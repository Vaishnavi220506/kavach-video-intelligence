"""Vercel adapter for the existing KAVACH FastAPI application.

Vercel Services removes the ``/api`` service prefix before invoking the
ASGI application. The existing local API intentionally keeps its ``/api``
routes, so this adapter restores that prefix at the service boundary.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.main import app as local_app


class ApiPrefixAdapter:
    """Restore the local API prefix after Vercel service routing."""

    def __init__(self, app: Callable[..., Awaitable[Any]]) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> Any:
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        adapted_scope = dict(scope)
        path = str(scope.get("path") or "/")
        if not path.startswith("/api"):
            adapted_scope["path"] = "/api" + (path if path.startswith("/") else f"/{path}")
            raw_path = scope.get("raw_path")
            if raw_path is not None:
                adapted_scope["raw_path"] = adapted_scope["path"].encode("utf-8")
        return await self.app(adapted_scope, receive, send)


app = ApiPrefixAdapter(local_app)
