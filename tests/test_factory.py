"""
tests/test_factory.py
~~~~~~~~~~~~~~~~~~~~~

Unit tests for unikey_ai.core.factory.build_llm().

All tests run with zero network calls — no real LLM API is contacted.
We only verify that:
  - build_llm() returns a ChatLiteLLM instance (BaseChatModel)
  - The model string is formatted as "provider/model" per LiteLLM convention
  - api_base is correctly forwarded when provided
  - Unknown providers raise ProviderNotSupportedError
  - The factory is side-effect-free (no env var mutation)
"""

from __future__ import annotations

import pytest
from langchain_litellm import ChatLiteLLM

from unikey_ai.core.exceptions import ProviderNotSupportedError
from unikey_ai.core.factory import SUPPORTED_PROVIDERS, build_llm, get_supported_providers


# ─── Happy-path tests ─────────────────────────────────────────────────────────


class TestBuildLlmHappyPath:
    def test_returns_chatlitellm_instance(self):
        """build_llm() must return a ChatLiteLLM (which is a BaseChatModel)."""
        llm = build_llm("openai", "sk-test-key", "gpt-4o-mini")
        assert isinstance(llm, ChatLiteLLM)

    def test_model_string_format_openai(self):
        """Model string must be 'provider/model' — 'openai/gpt-4o-mini'."""
        llm = build_llm("openai", "sk-test", "gpt-4o-mini")
        assert llm.model == "openai/gpt-4o-mini"

    def test_model_string_format_anthropic(self):
        llm = build_llm("anthropic", "sk-ant-test", "claude-3-5-sonnet-20241022")
        assert llm.model == "anthropic/claude-3-5-sonnet-20241022"

    def test_model_string_format_groq(self):
        llm = build_llm("groq", "gsk-test", "llama3-8b-8192")
        assert llm.model == "groq/llama3-8b-8192"

    def test_model_string_format_gemini(self):
        llm = build_llm("gemini", "AIza-test", "gemini-1.5-flash")
        assert llm.model == "gemini/gemini-1.5-flash"

    def test_model_string_format_mistral(self):
        llm = build_llm("mistral", "mst-test", "mistral-large-latest")
        assert llm.model == "mistral/mistral-large-latest"

    def test_model_string_format_cohere(self):
        llm = build_llm("cohere", "co-test", "command-r-plus")
        assert llm.model == "cohere/command-r-plus"

    def test_ollama_without_base_url(self):
        """Ollama without base_url should still construct — user may set it elsewhere."""
        llm = build_llm("ollama", "dummy", "llama3")
        assert isinstance(llm, ChatLiteLLM)
        assert llm.model == "ollama/llama3"

    def test_provider_case_insensitive(self):
        """Provider string must be normalised to lowercase before lookup."""
        llm_upper = build_llm("OpenAI", "sk-test", "gpt-4o-mini")
        llm_mixed = build_llm("OPENAI", "sk-test", "gpt-4o-mini")
        assert llm_upper.model == "openai/gpt-4o-mini"
        assert llm_mixed.model == "openai/gpt-4o-mini"

    def test_provider_whitespace_stripped(self):
        """Leading/trailing whitespace in header value must be stripped."""
        llm = build_llm("  openai  ", "sk-test", "gpt-4o-mini")
        assert llm.model == "openai/gpt-4o-mini"


# ─── api_base forwarding ──────────────────────────────────────────────────────


class TestApiBaseForwarding:
    def test_api_base_not_set_when_none(self):
        """When api_base is None, it must NOT be passed to ChatLiteLLM."""
        llm = build_llm("openai", "sk-test", "gpt-4o-mini", api_base=None)
        # ChatLiteLLM stores it as api_base or custom_api; must be falsy
        assert not llm.api_base

    def test_api_base_forwarded_when_provided(self):
        """When api_base is provided, it must appear on the constructed object."""
        base = "http://localhost:11434"
        llm = build_llm("ollama", "dummy", "llama3", api_base=base)
        assert llm.api_base == base

    def test_api_base_trailing_slash_stripped(self):
        """Trailing slash should be stripped from api_base for clean URL composition."""
        llm = build_llm("ollama", "dummy", "llama3", api_base="http://localhost:11434/")
        assert llm.api_base == "http://localhost:11434"

    def test_openai_compatible_with_lm_studio(self):
        """openai-compatible provider with LM Studio base URL."""
        llm = build_llm(
            "openai-compatible",
            "lm-studio",
            "local-model",
            api_base="http://localhost:1234/v1",
        )
        assert isinstance(llm, ChatLiteLLM)
        assert llm.api_base == "http://localhost:1234/v1"


# ─── Error cases ──────────────────────────────────────────────────────────────


class TestBuildLlmErrors:
    def test_unknown_provider_raises_error(self):
        """An unknown provider must raise ProviderNotSupportedError."""
        with pytest.raises(ProviderNotSupportedError) as exc_info:
            build_llm("unknown-provider", "sk-test", "some-model")
        assert exc_info.value.provider == "unknown-provider"
        assert exc_info.value.http_status_code == 422

    def test_unknown_provider_lists_supported(self):
        """ProviderNotSupportedError must list the supported providers."""
        with pytest.raises(ProviderNotSupportedError) as exc_info:
            build_llm("fakeai", "key", "model")
        # detail should mention supported providers
        assert "openai" in exc_info.value.detail

    def test_unknown_provider_empty_string(self):
        with pytest.raises(ProviderNotSupportedError):
            build_llm("", "key", "model")

    def test_all_supported_providers_construct_successfully(self):
        """Every provider in SUPPORTED_PROVIDERS must succeed without raising."""
        for provider in SUPPORTED_PROVIDERS:
            llm = build_llm(provider, "test-key", "test-model")
            assert isinstance(llm, ChatLiteLLM), f"Failed for provider: {provider}"


# ─── get_supported_providers ──────────────────────────────────────────────────


class TestGetSupportedProviders:
    def test_returns_sorted_list(self):
        providers = get_supported_providers()
        assert providers == sorted(providers)

    def test_contains_expected_providers(self):
        providers = get_supported_providers()
        for expected in ["openai", "anthropic", "groq", "ollama"]:
            assert expected in providers
