"""Protect the OpenAI-compatible HTTP surface using the offline echo backend."""

import json

from fastapi.testclient import TestClient

from kumaru.core.config import BackendConfig, KumaruConfig, ServerConfig
from kumaru.server.app import create_app


def chat_payload(stream: bool = False) -> dict:
    return {"model": "echo", "messages": [{"role": "user", "content": "hello"}], "stream": stream}


def test_get_healthz(test_client: TestClient) -> None:
    response = test_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "echo"
    assert body["loaded"] is True


def test_get_models(test_client: TestClient) -> None:
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "echo"


def test_get_info(test_client: TestClient) -> None:
    response = test_client.get("/v1/info")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["config"]["backend"]["name"] == "echo"


def test_get_metrics(test_client: TestClient) -> None:
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "requests" in response.json()


def test_chat_completion_non_streaming_returns_openai_shape(test_client: TestClient) -> None:
    response = test_client.post("/v1/chat/completions", json=chat_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert "You said: hello" in body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_chat_completion_streaming_returns_sse_frames(test_client: TestClient) -> None:
    with test_client.stream("POST", "/v1/chat/completions", json=chat_payload(stream=True)) as response:
        assert response.status_code == 200
        text = response.read().decode("utf-8")
    frames = [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    payloads = [json.loads(frame) for frame in frames[:-1]]
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert any("content" in payload["choices"][0]["delta"] for payload in payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_empty_messages_returns_error_status(test_client: TestClient) -> None:
    response = test_client.post("/v1/chat/completions", json={"model": "echo", "messages": []})
    assert response.status_code >= 400
    assert "error" in response.json()


def test_api_key_required_for_chat_completion() -> None:
    cfg = KumaruConfig(
        backend=BackendConfig(name="echo", model="echo", extra={"delay_s": 0}),
        server=ServerConfig(api_key="secret"),
    )
    with TestClient(create_app(cfg)) as client:
        missing = client.post("/v1/chat/completions", json=chat_payload())
        assert missing.status_code == 500
        ok = client.post(
            "/v1/chat/completions",
            json=chat_payload(),
            headers={"Authorization": "Bearer secret"},
        )
        assert ok.status_code == 200
