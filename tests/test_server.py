"""HTTP layer: chat, streaming, history and health."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kumaru.agent.chat import ChatAgent
from kumaru.core.errors import BackendError
from kumaru.server.app import create_app


@pytest.fixture
def client(config, agent: ChatAgent):
    with TestClient(create_app(config, agent=agent)) as test_client:
        yield test_client


def test_health_reports_the_backend(client) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "echo"


def test_config_endpoint_never_exposes_the_api_key(client, monkeypatch) -> None:
    monkeypatch.setenv("KUMARU_API_KEY", "super-secret-value")
    body = client.get("/api/config").json()
    assert body["backend"]["name"] == "echo"
    # Only the *name* of the env var is published, never its value.
    assert body["backend"]["api_key_env"] == "KUMARU_API_KEY"
    assert "super-secret-value" not in json.dumps(body)


def test_chat_returns_a_reply(client) -> None:
    response = client.post("/api/chat", json={"message": "hello", "session_id": "s"})
    assert response.status_code == 200
    assert "hello" in response.json()["reply"]


def test_chat_rejects_an_empty_message(client) -> None:
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_stream_emits_sse_frames(client) -> None:
    response = client.post(
        "/api/chat/stream", json={"message": "stream me", "session_id": "s"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    text, done = "", False
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        event = json.loads(line[5:])
        text += event.get("delta", "")
        done = done or event.get("done", False)
    assert "stream me" in text
    assert done is True


def test_history_round_trip_and_clear(client) -> None:
    client.post("/api/chat", json={"message": "remember this", "session_id": "s"})
    body = client.get("/api/history/s").json()
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]

    client.delete("/api/history/s")
    assert client.get("/api/history/s").json()["messages"] == []


def test_backend_failure_becomes_a_json_error(config, agent: ChatAgent) -> None:
    def boom(*_args, **_kwargs):
        raise BackendError("model exploded", hint="check the logs")

    agent.backend.generate = boom  # type: ignore[method-assign]
    with TestClient(create_app(config, agent=agent)) as client:
        response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    body = response.json()
    assert body["error"] == "BackendError"
    assert body["hint"] == "check the logs"


def test_stream_failure_is_delivered_as_an_sse_error(config, agent: ChatAgent) -> None:
    def boom(*_args, **_kwargs):
        raise BackendError("stream exploded")

    agent.backend.stream = boom  # type: ignore[method-assign]
    with TestClient(create_app(config, agent=agent)) as client:
        response = client.post("/api/chat/stream", json={"message": "hi"})
    assert "stream exploded" in response.text
