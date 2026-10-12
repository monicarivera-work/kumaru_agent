"""HTTP layer: an OpenAI-compatible API plus the web UI.

Compatibility is the feature. Because `/v1/chat/completions` matches the OpenAI
schema, the official `openai` SDK, LangChain, LlamaIndex, Continue.dev, and most
chat frontends can talk to Kumaru with a changed `base_url` and nothing else.
You inherit an ecosystem instead of building one.
"""

from kumaru.server.app import create_app

__all__ = ["create_app"]
