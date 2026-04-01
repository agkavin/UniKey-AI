"""
unikey_ai.core.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~

Custom exception hierarchy for UniKey-AI.

Each exception maps directly to an HTTP status code so that BYOKMiddleware
can convert them to proper JSON error responses without the developer writing
any try/except blocks.

Hierarchy:
    UniKeyError (base)
    ├── MissingHeaderError        → HTTP 400
    ├── InvalidKeyError           → HTTP 401
    ├── ProviderNotSupportedError → HTTP 422
    └── ProviderRateLimitError    → HTTP 429
"""

from __future__ import annotations


class UniKeyError(Exception):
    """
    Base exception for all UniKey-AI errors.

    Attributes:
        http_status_code: The HTTP status code this error maps to.
        detail: Human-readable error message safe to send to clients.
    """

    http_status_code: int = 500

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(status={self.http_status_code}, detail={self.detail!r})"


class MissingHeaderError(UniKeyError):
    """
    Raised when one or more required x-ai-* headers are absent from the request.

    HTTP 400 Bad Request — the client sent an incomplete request.

    Example:
        raise MissingHeaderError("x-ai-provider")
    """

    http_status_code: int = 400

    def __init__(self, header_name: str) -> None:
        self.header_name = header_name
        super().__init__(
            detail=f"Required header '{header_name}' is missing. "
            f"Please include '{header_name}' in your request headers."
        )


class InvalidKeyError(UniKeyError):
    """
    Raised when the provider rejects the supplied API key.

    HTTP 401 Unauthorized — the supplied credentials are invalid.

    This is typically caught from litellm.AuthenticationError and re-raised
    as a UniKey-native type by the middleware for uniform handling.
    """

    http_status_code: int = 401

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider
        provider_str = f" for provider '{provider}'" if provider else ""
        super().__init__(
            detail=f"The API key supplied{provider_str} was rejected. "
            "Please verify your key and try again."
        )


class ProviderNotSupportedError(UniKeyError):
    """
    Raised when the value of x-ai-provider is not in UniKey-AI's known set.

    HTTP 422 Unprocessable Entity — the request is syntactically valid but
    semantically incorrect (the provider value is not understood).
    """

    http_status_code: int = 422

    def __init__(self, provider: str, supported: list[str]) -> None:
        self.provider = provider
        self.supported = supported
        super().__init__(
            detail=f"Provider '{provider}' is not supported. "
            f"Supported providers: {', '.join(supported)}."
        )


class ProviderRateLimitError(UniKeyError):
    """
    Raised when the LLM provider returns a rate-limit response.

    HTTP 429 Too Many Requests — the user's key has been rate-limited.
    """

    http_status_code: int = 429

    def __init__(self, provider: str | None = None) -> None:
        self.provider = provider
        provider_str = f" ({provider})" if provider else ""
        super().__init__(
            detail=f"Rate limit reached{provider_str}. "
            "Please wait before retrying, or upgrade your API plan."
        )
