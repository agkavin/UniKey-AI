"""
unikey_ai.core.factory
~~~~~~~~~~~~~~~~~~~~~~

The single place in the library where ChatLiteLLM is instantiated.

Public API:
    build_llm(provider, api_key, model, api_base) -> ChatLiteLLM

Supported providers (Phase 1):
    openai, anthropic, gemini, groq, cohere, mistral, ollama,
    openai-compatible

Design decisions:
    - The model string is formatted as "{provider}/{model}" per LiteLLM's
      routing convention (e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022").
    - api_base is only passed when explicitly provided, which enables local
      model servers (Ollama, LM Studio, vLLM) without affecting cloud providers.
    - api_key is always passed via the constructor; it is NEVER stored or
      written to environment variables, keeping the library side-effect-free.
    - custom_llm_provider is passed explicitly so LiteLLM does not have to
      infer the provider from the model string — making routing deterministic.
"""

from __future__ import annotations

from typing import Optional

from langchain_litellm import ChatLiteLLM

from unikey_ai.core.exceptions import ProviderNotSupportedError

# ─── Supported providers ───────────────────────────────────────────────────────
#
# The keys in this dict are the normalized (lowercased) values accepted for the
# x-ai-provider header. The values are the provider strings that LiteLLM
# expects in its `custom_llm_provider` parameter.
#
# "openai-compatible" is a catch-all for any server exposing an OpenAI-
# compatible REST API (e.g. vLLM, LM Studio, Together AI, Anyscale, etc.).
# When used, x-ai-base-url MUST be provided.

SUPPORTED_PROVIDERS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "groq": "groq",
    "cohere": "cohere",
    "mistral": "mistral",
    "ollama": "ollama",
    "openai-compatible": "openai",  # Routes as OpenAI, but uses the custom base URL
}


def build_llm(
    provider: str,
    api_key: str,
    model: str,
    api_base: Optional[str] = None,
) -> ChatLiteLLM:
    """
    Build and return a configured ChatLiteLLM (BaseChatModel) instance.

    This is the only place in the library that constructs an LLM object.
    The returned object is a first-class LangChain BaseChatModel — it works
    with .invoke(), .stream(), .ainvoke(), .bind_tools(), LCEL pipes,
    LangGraph nodes, and CrewAI agents with no extra setup.

    Args:
        provider:  Normalized provider name (e.g. "openai", "anthropic", "ollama").
                   Must be one of the keys in SUPPORTED_PROVIDERS.
        api_key:   The user's API key for the provider. Pass "dummy" or any
                   non-empty string for local providers that don't require auth.
        model:     The model identifier (e.g. "gpt-4o", "llama3", "claude-3-5-sonnet-20241022").
                   This is combined with provider to form the LiteLLM model string.
        api_base:  Optional. Override the provider's default API base URL.
                   Required for local models (Ollama: http://localhost:11434,
                   LM Studio: http://localhost:1234/v1, vLLM, etc.)

    Returns:
        A fully configured ChatLiteLLM instance ready for immediate use.

    Raises:
        ProviderNotSupportedError: If `provider` is not in SUPPORTED_PROVIDERS.

    Examples:
        # Cloud provider
        llm = build_llm("openai", "sk-abc123", "gpt-4o")
        response = llm.invoke("Hello!")

        # Local Ollama model
        llm = build_llm("ollama", "dummy", "llama3", "http://localhost:11434")
        response = llm.invoke("Hello!")

        # OpenAI-compatible server (LM Studio)
        llm = build_llm(
            "openai-compatible", "lm-studio", "local-model",
            "http://localhost:1234/v1"
        )
    """
    normalized_provider = provider.strip().lower()

    if normalized_provider not in SUPPORTED_PROVIDERS:
        raise ProviderNotSupportedError(
            provider=provider,
            supported=sorted(SUPPORTED_PROVIDERS.keys()),
        )

    litellm_provider = SUPPORTED_PROVIDERS[normalized_provider]

    # LiteLLM routing convention: "provider/model"
    # e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022", "ollama/llama3"
    model_string = f"{litellm_provider}/{model}"

    kwargs: dict = {
        "model": model_string,
        "api_key": api_key,
        "custom_llm_provider": litellm_provider,
    }

    # Only inject api_base if explicitly provided — avoids overriding provider
    # defaults for cloud providers where base_url is not user-configurable.
    if api_base:
        kwargs["api_base"] = api_base.rstrip("/")

    return ChatLiteLLM(**kwargs)


def get_supported_providers() -> list[str]:
    """Return a sorted list of provider names accepted by x-ai-provider."""
    return sorted(SUPPORTED_PROVIDERS.keys())
