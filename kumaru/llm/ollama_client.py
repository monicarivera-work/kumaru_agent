"""
kumaru/llm/ollama_client.py
---------------------------
Native Ollama implementation of the BaseLLMClient interface.

We talk directly to Ollama's ``/api/chat`` endpoint so local models can be the
default runtime without depending on OpenAI-compatible shims.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import requests

from kumaru.config import LLMConfig
from kumaru.llm.base import BaseLLMClient, LLMResponse, Message


class OllamaClient(BaseLLMClient):
    """Wraps the Ollama chat API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._base_url = config.base_url.rstrip("/")

    def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        response = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._config.timeout,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})

        return LLMResponse(
            content=message.get("content"),
            tool_calls=self._normalise_tool_calls(message.get("tool_calls")),
            usage=self._extract_usage(data),
            raw=data,
        )

    def _normalise_tool_calls(
        self,
        tool_calls: Optional[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        normalised: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls or [], start=1):
            function = tool_call.get("function", {})
            arguments = function.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)

            normalised.append(
                {
                    "id": tool_call.get("id") or f"ollama_call_{index}",
                    "type": tool_call.get("type", "function"),
                    "function": {
                        "name": function["name"],
                        "arguments": arguments,
                    },
                }
            )
        return normalised

    def _extract_usage(self, data: dict[str, Any]) -> dict[str, int]:
        usage: dict[str, int] = {}

        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        if isinstance(prompt_tokens, int):
            usage["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            usage["completion_tokens"] = completion_tokens
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )

        return usage
