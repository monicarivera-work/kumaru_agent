"""HTTP surface: a FastAPI app plus the static browser UI it serves."""

from kumaru.server.app import create_app

__all__ = ["create_app"]
