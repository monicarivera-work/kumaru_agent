from __future__ import annotations

from typing import Any

from kumaru.config import LLMConfig
from kumaru.llm.base import Message
from kumaru.llm.ollama_client import OllamaClient


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_chat_uses_ollama_api_and_normalises_tool_calls(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: int):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse(
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": {"expression": "6 * 7"},
                            }
                        }
                    ],
                },
                "prompt_eval_count": 11,
                "eval_count": 3,
            }
        )

    monkeypatch.setattr("kumaru.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(LLMConfig(model="llama3.1", base_url="http://localhost:11434/"))
    response = client.chat(
        messages=[Message(role="user", content="What is 6 * 7?")],
        tools=[{"type": "function", "function": {"name": "calculator"}}],
    )

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["json"]["model"] == "llama3.1"
    assert captured["json"]["tools"][0]["function"]["name"] == "calculator"
    assert response.content is None
    assert response.tool_calls[0]["function"]["arguments"] == '{"expression": "6 * 7"}'
    assert response.usage == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }


def test_chat_returns_plain_text_response(monkeypatch):
    def fake_post(url: str, json: dict[str, Any], timeout: int):
        return DummyResponse({"message": {"content": "Hello from Ollama"}})

    monkeypatch.setattr("kumaru.llm.ollama_client.requests.post", fake_post)

    client = OllamaClient(LLMConfig())
    response = client.chat(messages=[Message(role="user", content="Hello")])

    assert response.content == "Hello from Ollama"
    assert response.tool_calls == []
    assert response.usage == {}
