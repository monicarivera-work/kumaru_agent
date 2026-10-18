"""Protect the OpenAI-compatible HTTP surface using the offline echo backend."""

import json

import pytest
from fastapi.testclient import TestClient

from kumaru.core.config import BackendConfig, KumaruConfig, ServerConfig, ThinkingModeConfig
from kumaru.server.app import create_app
from kumaru.server.schemas import ChatCompletionRequest

ECHO_THOUGHT = (
    "The user asked: hello. Echo cannot reason; this text only exercises the reasoning channel."
)


def chat_payload(stream: bool = False) -> dict:
    return {"model": "echo", "messages": [{"role": "user", "content": "hello"}], "stream": stream}


def sse_payloads(client: TestClient, payload: dict) -> list[dict]:
    """POST a streaming request; return every JSON event before `[DONE]`."""
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        text = response.read().decode("utf-8")
    frames = [line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    return [json.loads(frame) for frame in frames[:-1]]


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


def test_empty_messages_returns_400(test_client: TestClient) -> None:
    response = test_client.post("/v1/chat/completions", json={"model": "echo", "messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "RequestError"


def test_every_client_system_message_reaches_the_model(
    test_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert test_client.post("/v1/chat/completions", json=chat_payload()).status_code == 200  # loaded
    backend = test_client.app.state.kumaru.backend
    prompts: list[list] = []
    stream = backend.stream

    def spy(messages, params):
        prompts.append(list(messages))
        return stream(messages, params)

    monkeypatch.setattr(backend, "stream", spy)
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "Use metric units."},
    ]
    response = test_client.post("/v1/chat/completions", json={**chat_payload(), "messages": messages})
    assert response.status_code == 200
    [prompt] = prompts
    assert [m.role.value for m in prompt] == ["system", "user"]
    assert prompt[0].content.endswith("\n\nBe terse.\n\nUse metric units.")  # after Kumaru's own


def test_api_key_required_for_chat_completion() -> None:
    cfg = KumaruConfig(
        backend=BackendConfig(name="echo", model="echo", extra={"delay_s": 0}),
        server=ServerConfig(api_key="secret"),
    )
    with TestClient(create_app(cfg)) as client:
        missing = client.post("/v1/chat/completions", json=chat_payload())
        assert missing.status_code == 401
        assert missing.json()["error"]["type"] == "AuthError"
        ok = client.post(
            "/v1/chat/completions",
            json=chat_payload(),
            headers={"Authorization": "Bearer secret"},
        )
        assert ok.status_code == 200


# -- statistics and thinking -----------------------------------------------------------


def test_streaming_final_frame_carries_timing_stats(test_client: TestClient) -> None:
    final = sse_payloads(test_client, chat_payload(stream=True))[-1]
    assert final["choices"][0]["delta"] == {}
    stats = final["kumaru"]  # the UI's tok/s and timing line
    assert set(stats) == {
        "latency_ms", "time_to_first_token_ms", "thinking_ms",
        "prompt_tokens", "completion_tokens", "tokens_per_second",
    }  # fmt: skip
    assert stats["thinking_ms"] is None  # nothing was thought
    assert stats["completion_tokens"] > 0


def test_streamed_thoughts_arrive_before_the_answer(test_client: TestClient) -> None:
    payloads = sse_payloads(test_client, {**chat_payload(stream=True), "thinking": True})
    deltas = [payload["choices"][0]["delta"] for payload in payloads[1:-1]]
    first_answer = next(i for i, delta in enumerate(deltas) if "content" in delta)
    assert first_answer > 0
    assert all(set(delta) == {"reasoning_content"} for delta in deltas[:first_answer])
    assert all(set(delta) == {"content"} for delta in deltas[first_answer:])
    assert "".join(delta["reasoning_content"] for delta in deltas[:first_answer]) == ECHO_THOUGHT
    stats = payloads[-1]["kumaru"]
    assert stats["completion_tokens"] == len(deltas)  # echo counts one token per frame
    assert stats["thinking_ms"] is not None


def test_non_streaming_reply_carries_reasoning_only_when_thinking(test_client: TestClient) -> None:
    plain = test_client.post("/v1/chat/completions", json=chat_payload()).json()
    assert "reasoning_content" not in plain["choices"][0]["message"]
    thought = test_client.post(
        "/v1/chat/completions", json={**chat_payload(), "thinking": True}
    ).json()
    message = thought["choices"][0]["message"]
    assert message["reasoning_content"] == ECHO_THOUGHT
    assert message["content"] == plain["choices"][0]["message"]["content"]


def test_thinking_budget_reaches_the_backend(test_client: TestClient) -> None:
    body = {**chat_payload(), "thinking": True, "thinking_budget": 2}
    message = test_client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
    assert message["reasoning_content"] == "The user asked: "  # two 8-character pieces


@pytest.mark.parametrize("stream", [False, True])
def test_a_thought_that_fills_max_tokens_ends_with_length_and_no_answer(
    test_client: TestClient, stream: bool
) -> None:
    body = {**chat_payload(stream=stream), "thinking": True, "max_tokens": 5}
    if stream:
        payloads = sse_payloads(test_client, body)
        deltas = [payload["choices"][0]["delta"] for payload in payloads[1:-1]]
        assert all(set(delta) == {"reasoning_content"} for delta in deltas)
        reasoning = "".join(delta["reasoning_content"] for delta in deltas)
        finish = payloads[-1]["choices"][0]["finish_reason"]
        completion_tokens = payloads[-1]["kumaru"]["completion_tokens"]
    else:
        reply = test_client.post("/v1/chat/completions", json=body).json()
        message = reply["choices"][0]["message"]
        assert not message["content"]
        reasoning = message["reasoning_content"]
        finish = reply["choices"][0]["finish_reason"]
        completion_tokens = reply["usage"]["completion_tokens"]
    assert reasoning == ECHO_THOUGHT[:40]  # five 8-character pieces, then the cap
    assert (finish, completion_tokens) == ("length", 5)


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({}, None),  # use the server default
        ({"thinking": False, "reasoning_effort": "high"}, False),  # Kumaru's own field wins
        ({"chat_template_kwargs": {"enable_thinking": True}}, True),  # vLLM / SGLang
        ({"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high"}, False),
        ({"chat_template_kwargs": {"enable_thinking": "false"}}, False),  # a string, not a bool
        ({"chat_template_kwargs": {"enable_thinking": None}}, None),
        ({"chat_template_kwargs": {"other": 1}}, None),
        ({"reasoning_effort": "low"}, True),  # OpenAI
        ({"reasoning_effort": " None "}, False),
        ({"reasoning_effort": ""}, None),
    ],
)
def test_resolved_thinking_accepts_three_spellings(fields: dict, expected: bool | None) -> None:
    request = ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}], **fields)
    assert request.resolved_thinking() is expected


def test_info_reports_thinking_mode_without_leaking_the_api_key() -> None:
    cfg = KumaruConfig(
        backend=BackendConfig(
            name="echo", model="echo", api_key="not-for-the-ui", extra={"delay_s": 0}
        ),
        thinking_mode=ThinkingModeConfig(max_tokens=4000, thinking_budget=1000),
    )
    with TestClient(create_app(cfg)) as client:
        response = client.get("/v1/info")
    assert response.status_code == 200
    config = response.json()["config"]
    assert config["thinking_mode"]["max_tokens"] == 4000
    assert config["thinking_mode"]["thinking_budget"] == 1000
    assert "api_key" not in config["backend"]
    assert "not-for-the-ui" not in response.text


def test_ui_assets_are_served_with_browser_safe_types(test_client: TestClient) -> None:
    # Browsers refuse a stylesheet served as text/plain, which the Windows
    # registry can otherwise cause; app.py pins these types.
    expected = {"/": "text/html", "/static/app.js": "text/javascript", "/static/styles.css": "text/css"}
    for path, mime in expected.items():
        response = test_client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-type"].split(";")[0] == mime, path
