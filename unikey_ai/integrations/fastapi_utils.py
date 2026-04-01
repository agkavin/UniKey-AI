"""
unikey_ai.integrations.fastapi_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FastAPI integration layer: middleware and dependency injection.

Public API:
    BYOKMiddleware  — Starlette BaseHTTPMiddleware that catches all LiteLLM
                      and UniKey-AI errors and converts them to clean JSON responses.
    get_byok_llm()  — Returns a FastAPI Depends-compatible callable that extracts
                      x-ai-* headers, builds a ChatLiteLLM, and yields it.

Usage:
    from fastapi import FastAPI, Depends
    from unikey_ai import BYOKMiddleware, get_byok_llm

    app = FastAPI()
    app.add_middleware(BYOKMiddleware)

    @app.post("/chat")
    async def chat(prompt: str, llm=Depends(get_byok_llm())):
        response = await llm.ainvoke(prompt)
        return {"result": response.content}
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from typing import Optional

import litellm
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from unikey_ai.core.exceptions import (
    MissingHeaderError,
    ProviderNotSupportedError,
    UniKeyError,
)
from unikey_ai.core.factory import build_llm

logger = logging.getLogger(__name__)

# ─── Header name constants ────────────────────────────────────────────────────

HEADER_PROVIDER = "x-ai-provider"
HEADER_KEY = "x-ai-key"
HEADER_MODEL = "x-ai-model"
HEADER_BASE_URL = "x-ai-base-url"


# ─── Middleware ───────────────────────────────────────────────────────────────


class BYOKMiddleware(BaseHTTPMiddleware):
    """
    Global BYOK error-catching middleware for FastAPI.

    Place this as the outermost middleware so it wraps every route, including
    routes that use get_byok_llm(). It catches:

        - UniKeyError subclasses (MissingHeaderError, ProviderNotSupportedError,
          InvalidKeyError, ProviderRateLimitError) → their mapped HTTP status codes
        - litellm.AuthenticationError → 401 Unauthorized
        - litellm.RateLimitError      → 429 Too Many Requests
        - litellm.BadRequestError     → 400 Bad Request
        - litellm.NotFoundError       → 404 Not Found (model not found on provider)
        - Any other Exception         → 500 Internal Server Error (no stack trace leaked)

    The developer writes zero try/except blocks in their route handlers.

    Registration:
        app.add_middleware(BYOKMiddleware)
    """

    def __init__(self, app: ASGIApp, *, include_error_detail: bool = True) -> None:
        """
        Args:
            app: The ASGI application (injected by FastAPI automatically).
            include_error_detail: If True, include a human-readable 'detail' field
                in error responses. Set to False in production if you want fully
                opaque errors for security hardening. Default: True.
        """
        super().__init__(app)
        self.include_error_detail = include_error_detail

    async def dispatch(self, request: Request, call_next: Callable):  # type: ignore[override]
        try:
            response = await call_next(request)
            return response

        # ── UniKey-AI own exceptions ──────────────────────────────────────────
        except UniKeyError as exc:
            logger.warning(
                "UniKeyError during request [%s %s]: %s",
                request.method,
                request.url.path,
                exc.detail,
            )
            return self._error_response(exc.http_status_code, exc.detail)

        # ── LiteLLM / provider authentication failure ─────────────────────────
        except litellm.AuthenticationError as exc:
            logger.warning(
                "AuthenticationError during request [%s %s] — provider=%s",
                request.method,
                request.url.path,
                getattr(exc, "llm_provider", "unknown"),
            )
            detail = (
                f"Authentication failed with provider "
                f"'{getattr(exc, 'llm_provider', 'unknown')}'. "
                "Please verify your API key."
            )
            return self._error_response(401, detail)

        # ── LiteLLM rate limit ────────────────────────────────────────────────
        except litellm.RateLimitError as exc:
            logger.warning(
                "RateLimitError during request [%s %s] — provider=%s",
                request.method,
                request.url.path,
                getattr(exc, "llm_provider", "unknown"),
            )
            detail = (
                f"Rate limit exceeded for provider "
                f"'{getattr(exc, 'llm_provider', 'unknown')}'. "
                "Please wait and retry."
            )
            return self._error_response(429, detail)

        # ── LiteLLM bad request (e.g. invalid model name) ─────────────────────
        except litellm.BadRequestError as exc:
            logger.warning(
                "BadRequestError during request [%s %s]: %s",
                request.method,
                request.url.path,
                str(exc),
            )
            return self._error_response(400, str(exc))

        # ── LiteLLM 404 (model not found on the provider) ─────────────────────
        except litellm.NotFoundError as exc:
            logger.warning(
                "NotFoundError (model not found) during request [%s %s]: %s",
                request.method,
                request.url.path,
                str(exc),
            )
            return self._error_response(404, str(exc))

        # ── Catch-all: unknown / unexpected errors ────────────────────────────
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unhandled exception during request [%s %s]",
                request.method,
                request.url.path,
            )
            # Never leak internal stack traces to the client.
            detail = "An unexpected internal error occurred." if self.include_error_detail else "Internal Server Error"
            return self._error_response(500, detail)

    def _error_response(self, status_code: int, detail: str) -> JSONResponse:
        """Build a standardised JSON error response."""
        content: dict = {"error": True, "detail": detail, "status_code": status_code}
        return JSONResponse(status_code=status_code, content=content)


# ─── Dependency ───────────────────────────────────────────────────────────────


def get_byok_llm(
    *,
    require_base_url_for: Optional[list[str]] = None,
) -> Callable:
    """
    FastAPI dependency factory for BYOK LLM injection.

    Returns a dependency callable that, when used with Depends(), extracts
    x-ai-* headers from the current request, validates them, constructs a
    ChatLiteLLM (BaseChatModel), and yields it to the route handler.

    The yielded object is a first-class LangChain BaseChatModel — it supports
    .invoke(), .stream(), .ainvoke(), .bind_tools(), LCEL chains, LangGraph,
    and CrewAI with no extra configuration.

    Args:
        require_base_url_for: Optional list of provider names that MUST supply
            x-ai-base-url (e.g. ["ollama", "openai-compatible"]). If a request
            uses one of these providers without a base URL, a MissingHeaderError
            is raised. Defaults to None (no enforcement).

    Usage:
        @app.post("/chat")
        async def chat(prompt: str, llm=Depends(get_byok_llm())):
            return {"result": (await llm.ainvoke(prompt)).content}

        # Enforce base URL for local providers:
        @app.post("/local")
        async def local(llm=Depends(get_byok_llm(require_base_url_for=["ollama"]))):
            ...

    Raises:
        MissingHeaderError:        When a required header is absent.
        ProviderNotSupportedError: When x-ai-provider is not recognised.
    """
    _require_base_url_for: list[str] = (
        [p.lower() for p in require_base_url_for] if require_base_url_for else []
    )

    async def _dependency(request: Request) -> AsyncGenerator:
        # ── Extract required headers ──────────────────────────────────────────
        provider = request.headers.get(HEADER_PROVIDER)
        api_key = request.headers.get(HEADER_KEY)
        model = request.headers.get(HEADER_MODEL)

        if not provider:
            raise MissingHeaderError(HEADER_PROVIDER)
        if not api_key:
            raise MissingHeaderError(HEADER_KEY)
        if not model:
            raise MissingHeaderError(HEADER_MODEL)

        # ── Extract optional base URL ──────────────────────────────────────────
        api_base: Optional[str] = request.headers.get(HEADER_BASE_URL) or None

        # ── Enforce base URL requirement for specific providers ────────────────
        if _require_base_url_for and provider.strip().lower() in _require_base_url_for:
            if not api_base:
                raise MissingHeaderError(HEADER_BASE_URL)

        # ── Build LLM — ProviderNotSupportedError raised here if invalid ───────
        llm = build_llm(
            provider=provider,
            api_key=api_key,
            model=model,
            api_base=api_base,
        )

        # Yield: the LLM is scoped to this single request lifecycle.
        # Nothing is stored, persisted, or shared across requests.
        yield llm

    return _dependency
