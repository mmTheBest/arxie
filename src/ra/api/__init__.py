"""FastAPI entrypoint for Academic Research Assistant."""

from ra.api.app import app, create_app

__all__ = ["app", "create_app"]
