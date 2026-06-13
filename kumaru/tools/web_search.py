"""
kumaru/tools/web_search.py
--------------------------
A web-search stub tool.

Why a stub?
  Real web search requires an API key (e.g. Serper, Brave, SerpAPI).
  This implementation shows the *structure* of a search tool and returns a
  realistic-looking stub response so you can develop and test the agent
  without spending money on search API credits.

  To make it real, replace the ``_stub_search`` call inside ``run()`` with
  an actual HTTP request to your search provider of choice.

Concept: Tool output formatting
---------------------------------
The model reads the tool's output as plain text.  Formatting matters:
  * Numbered lists help the model cite sources.
  * Keep results short – long tool output wastes context tokens.
  * Include the URL so the model can pass it on to the user.
"""

from __future__ import annotations

from typing import Any

from kumaru.tools.base import BaseTool, ToolError


class WebSearchTool(BaseTool):
    """Search the web and return the top results.

    Args:
        api_key: Your search provider API key.  Pass ``None`` (default) to
                 use the stub implementation.
    """

    name = "web_search"
    description = (
        "Search the internet for up-to-date information. "
        "Use this when the user asks about recent events, news, current data, "
        "or anything that might have changed after your training cut-off date."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return (1–5).",
                "default": 3,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    def run(self, query: str, num_results: int = 3, **_: Any) -> str:  # type: ignore[override]
        """Execute the search and return formatted results."""
        if not query.strip():
            raise ToolError("Search query cannot be empty.")

        num_results = max(1, min(5, num_results))

        if self._api_key:
            return self._live_search(query, num_results)
        return self._stub_search(query, num_results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stub_search(self, query: str, num_results: int) -> str:
        """Return a clearly-labelled stub response for development/testing."""
        results = [
            f"{i}. [STUB] Result for '{query}' – "
            f"https://example.com/result{i}"
            for i in range(1, num_results + 1)
        ]
        return (
            "[STUB – no real search performed]\n"
            + "\n".join(results)
        )

    def _live_search(self, query: str, num_results: int) -> str:
        """Call a real search API.

        Replace this with your chosen provider.  Example below uses Serper.dev.
        """
        try:
            import requests
        except ImportError as exc:
            raise ToolError(
                "The 'requests' package is needed for live search.\n"
                "Install it with:  pip install requests"
            ) from exc

        response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": self._api_key,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        organic = data.get("organic", [])
        if not organic:
            return "No results found."

        lines = []
        for i, item in enumerate(organic[:num_results], start=1):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append(f"{i}. {title}\n   {snippet}\n   {link}")
        return "\n\n".join(lines)
