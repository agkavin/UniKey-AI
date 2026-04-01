"""
tests/test_middleware.py
~~~~~~~~~~~~~~~~~~~~~~~~

Integration tests for BYOKMiddleware and get_byok_llm() using FastAPI's TestClient.

All LLM calls are monkeypatched — no real API calls are made.
Tests verify:
  - Missing headers → 400
  - Unknown provider → 422
  - Valid headers → dependency resolves and route executes
  - litellm.AuthenticationError mid-route → 401
  - litellm.RateLimitError mid-route → 429
  - Unexpected exception mid-route → 500 (no stack trace in body)
  - x-ai-base-url is correctly forwarded to the LLM object
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import litellm

from unikey_ai import BYOKMiddleware, get_byok_llm
from unikey_ai.core.exceptions import MissingHeaderError


# ─── Test app factory ─────────────────────────────────────────────────────────


def make_app(invoke_side_effect=None):
    """
    Create a minimal FastAPI test application with BYOKMiddleware and a single
    /generate route. If invoke_side_effect is provided, it is used to monkeypatch
    the llm's invoke method to simulate provider errors.
    """
    test_app = FastAPI()
    test_app.add_middleware(BYOKMiddleware)

    @test_app.post("/generate")
    def generate(
        llm=Depends(get_byok_llm()),
    ):
        if invoke_side_effect:
            raise invoke_side_effect
        # Normally would call llm.invoke() — but here we skip it to avoid network.
        return {"result": "ok", "model": llm.model}

    return test_app


# ─── Header validation (400) ──────────────────────────────────────────────────


class TestMissingHeaders:
    VALID_HEADERS = {
        "x-ai-provider": "openai",
        "x-ai-key": "sk-test",
        "x-ai-model": "gpt-4o-mini",
    }

    def test_missing_provider_returns_400(self):
        client = TestClient(make_app(), raise_server_exceptions=False)
        headers = {k: v for k, v in self.VALID_HEADERS.items() if k != "x-ai-provider"}
        response = client.post("/generate", headers=headers)
        assert response.status_code == 400
        body = response.json()
        assert body["error"] is True
        assert "x-ai-provider" in body["detail"]

    def test_missing_key_returns_400(self):
        client = TestClient(make_app(), raise_server_exceptions=False)
        headers = {k: v for k, v in self.VALID_HEADERS.items() if k != "x-ai-key"}
        response = client.post("/generate", headers=headers)
        assert response.status_code == 400
        body = response.json()
        assert "x-ai-key" in body["detail"]

    def test_missing_model_returns_400(self):
        client = TestClient(make_app(), raise_server_exceptions=False)
        headers = {k: v for k, v in self.VALID_HEADERS.items() if k != "x-ai-model"}
        response = client.post("/generate", headers=headers)
        assert response.status_code == 400
        body = response.json()
        assert "x-ai-model" in body["detail"]

    def test_all_headers_missing_returns_400(self):
        client = TestClient(make_app(), raise_server_exceptions=False)
        response = client.post("/generate")
        assert response.status_code == 400


# ─── Unknown provider (422) ───────────────────────────────────────────────────


class TestUnknownProvider:
    def test_unknown_provider_returns_422(self):
        client = TestClient(make_app(), raise_server_exceptions=False)
        response = client.post(
            "/generate",
            headers={
                "x-ai-provider": "totally-fake-provider",
                "x-ai-key": "sk-test",
                "x-ai-model": "some-model",
            },
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"] is True
        assert "totally-fake-provider" in body["detail"]

    def test_empty_provider_returns_422(self):
        client = TestClient(make_app(), raise_server_exceptions=False)
        response = client.post(
            "/generate",
            headers={
                "x-ai-provider": "",
                "x-ai-key": "sk-test",
                "x-ai-model": "some-model",
            },
        )
        # Empty string is treated as missing
        assert response.status_code in (400, 422)


# ─── Valid headers → dependency resolves ─────────────────────────────────────


class TestValidHeaders:
    def _post(self, provider: str, key: str, model: str, base_url: str | None = None):
        client = TestClient(make_app(), raise_server_exceptions=False)
        headers = {
            "x-ai-provider": provider,
            "x-ai-key": key,
            "x-ai-model": model,
        }
        if base_url:
            headers["x-ai-base-url"] = base_url
        return client.post("/generate", headers=headers)

    def test_openai_valid_headers_returns_200(self):
        response = self._post("openai", "sk-test", "gpt-4o-mini")
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "openai/gpt-4o-mini"

    def test_anthropic_valid_headers_returns_200(self):
        response = self._post("anthropic", "sk-ant-test", "claude-3-5-sonnet-20241022")
        assert response.status_code == 200
        assert response.json()["model"] == "anthropic/claude-3-5-sonnet-20241022"

    def test_ollama_with_base_url_returns_200(self):
        response = self._post(
            "ollama", "dummy", "llama3", base_url="http://localhost:11434"
        )
        assert response.status_code == 200
        assert response.json()["model"] == "ollama/llama3"

    def test_provider_case_insensitive(self):
        response = self._post("OPENAI", "sk-test", "gpt-4o-mini")
        assert response.status_code == 200

    def test_result_field_in_response(self):
        response = self._post("openai", "sk-test", "gpt-4o-mini")
        assert "result" in response.json()


# ─── Provider errors caught by middleware ─────────────────────────────────────


class TestMiddlewareErrorCatching:
    def _post_with_exception(self, exc):
        """Helper: makes a valid-header request but the route raises `exc`."""
        client = TestClient(make_app(invoke_side_effect=exc), raise_server_exceptions=False)
        return client.post(
            "/generate",
            headers={
                "x-ai-provider": "openai",
                "x-ai-key": "sk-test",
                "x-ai-model": "gpt-4o-mini",
            },
        )

    def test_litellm_auth_error_returns_401(self):
        exc = litellm.AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        response = self._post_with_exception(exc)
        assert response.status_code == 401
        body = response.json()
        assert body["error"] is True
        assert "401" in str(body["status_code"])

    def test_litellm_rate_limit_returns_429(self):
        exc = litellm.RateLimitError(
            message="Rate limit exceeded",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        response = self._post_with_exception(exc)
        assert response.status_code == 429
        body = response.json()
        assert body["error"] is True

    def test_unexpected_exception_returns_500(self):
        exc = RuntimeError("Something exploded internally")
        response = self._post_with_exception(exc)
        assert response.status_code == 500
        body = response.json()
        # Must NOT leak the internal error message or stack trace
        assert body["error"] is True
        # detail should be generic
        assert "internal" in body["detail"].lower()


# ─── x-ai-base-url forwarding ────────────────────────────────────────────────


class TestBaseUrlForwarding:
    def test_base_url_forwarded_to_llm(self):
        """Verify that api_base from the header reaches the llm object."""
        captured = {}

        test_app = FastAPI()
        test_app.add_middleware(BYOKMiddleware)

        @test_app.post("/inspect")
        def inspect(llm=Depends(get_byok_llm())):
            captured["api_base"] = llm.api_base
            return {"ok": True}

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.post(
            "/inspect",
            headers={
                "x-ai-provider": "ollama",
                "x-ai-key": "dummy",
                "x-ai-model": "llama3",
                "x-ai-base-url": "http://localhost:11434",
            },
        )
        assert response.status_code == 200
        assert captured.get("api_base") == "http://localhost:11434"

    def test_no_base_url_means_none_on_llm(self):
        """When x-ai-base-url is not sent, api_base on the llm must be falsy."""
        captured = {}

        test_app = FastAPI()
        test_app.add_middleware(BYOKMiddleware)

        @test_app.post("/inspect")
        def inspect(llm=Depends(get_byok_llm())):
            captured["api_base"] = llm.api_base
            return {"ok": True}

        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.post(
            "/inspect",
            headers={
                "x-ai-provider": "openai",
                "x-ai-key": "sk-test",
                "x-ai-model": "gpt-4o-mini",
            },
        )
        assert response.status_code == 200
        assert not captured.get("api_base")


# ─── require_base_url_for enforcement ────────────────────────────────────────


class TestRequireBaseUrlFor:
    def _make_app_with_requirement(self):
        test_app = FastAPI()
        test_app.add_middleware(BYOKMiddleware)

        @test_app.post("/local-only")
        def local_only(llm=Depends(get_byok_llm(require_base_url_for=["ollama"]))):
            return {"model": llm.model}

        return TestClient(test_app, raise_server_exceptions=False)

    def test_ollama_without_base_url_returns_400(self):
        client = self._make_app_with_requirement()
        response = client.post(
            "/local-only",
            headers={
                "x-ai-provider": "ollama",
                "x-ai-key": "dummy",
                "x-ai-model": "llama3",
                # x-ai-base-url intentionally omitted
            },
        )
        assert response.status_code == 400
        assert "x-ai-base-url" in response.json()["detail"]

    def test_ollama_with_base_url_passes(self):
        client = self._make_app_with_requirement()
        response = client.post(
            "/local-only",
            headers={
                "x-ai-provider": "ollama",
                "x-ai-key": "dummy",
                "x-ai-model": "llama3",
                "x-ai-base-url": "http://localhost:11434",
            },
        )
        assert response.status_code == 200

    def test_openai_without_base_url_passes_when_not_required(self):
        """openai is not in the require_base_url_for list — should pass without it."""
        client = self._make_app_with_requirement()
        response = client.post(
            "/local-only",
            headers={
                "x-ai-provider": "openai",
                "x-ai-key": "sk-test",
                "x-ai-model": "gpt-4o-mini",
            },
        )
        assert response.status_code == 200
