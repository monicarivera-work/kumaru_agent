"""
kumaru/config.py
----------------
Centralised configuration for the Kumaru agent.

Why a dedicated config module?
  In enterprise code you never want API keys or tuning parameters scattered
  across files.  A single place makes it easy to switch models, adjust
  temperature, or swap providers without hunting through the codebase.

The Config dataclass is intentionally plain Python – no framework magic –
so that you can understand exactly what each field does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# python-dotenv lets us keep secrets in a local .env file instead of
# hard-coding them.  It is a no-op when running in CI / production
# environments that inject environment variables another way.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; plain env vars still work


@dataclass
class LLMConfig:
    """Settings that control how we talk to the language model.

    Concept: LLM hyper-parameters
    ------------------------------
    * provider    – which backend to use (e.g. "ollama", "openai")
    * model       – which checkpoint to use (e.g. "llama3.1", "gpt-4o")
    * temperature – randomness of the output (0 = deterministic, 1 = creative)
    * max_tokens  – upper bound on the response length (cost control)
    * timeout     – seconds to wait for the API before giving up
    """

    provider: str = field(
        default_factory=lambda: os.getenv("KUMARU_LLM_PROVIDER", "ollama")
    )
    model: str = field(
        default_factory=lambda: os.getenv("KUMARU_LLM_MODEL", "llama3.1")
    )
    temperature: float = 0.0          # 0 keeps the agent's reasoning consistent
    max_tokens: int = 4096
    timeout: int = 60
    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )


@dataclass
class AgentConfig:
    """Top-level agent behaviour settings.

    Concept: Agent loop control
    ---------------------------
    * max_iterations – safety limit so the agent cannot loop forever
    * verbose        – print the agent's inner monologue for debugging
    * system_prompt  – the persona / instructions given to the LLM at the
                       start of every conversation
    """

    max_iterations: int = 10
    verbose: bool = False
    system_prompt: str = (
        "You are Kumaru, a helpful AI assistant. "
        "Think step-by-step before answering. "
        "When you need to use a tool, say so explicitly."
    )
    llm: LLMConfig = field(default_factory=LLMConfig)
