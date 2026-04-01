"""
UniKey-AI — BYOK Middleware & Dependency Injection for FastAPI.

Public API surface:

    from unikey_ai import BYOKMiddleware, get_byok_llm   # FastAPI integration
    from unikey_ai import build_llm                       # Raw factory (optional)
    from unikey_ai import (                               # Exceptions
        UniKeyError, MissingHeaderError,
        InvalidKeyError, ProviderNotSupportedError,
        ProviderRateLimitError,
    )
"""

from unikey_ai.core.exceptions import (
    UniKeyError,
    MissingHeaderError,
    InvalidKeyError,
    ProviderNotSupportedError,
    ProviderRateLimitError,
)
from unikey_ai.core.factory import build_llm
from unikey_ai.integrations.fastapi_utils import BYOKMiddleware, get_byok_llm

__version__ = "0.1.0"

__all__ = [
    # FastAPI integration
    "BYOKMiddleware",
    "get_byok_llm",
    # Factory
    "build_llm",
    # Exceptions
    "UniKeyError",
    "MissingHeaderError",
    "InvalidKeyError",
    "ProviderNotSupportedError",
    "ProviderRateLimitError",
]
