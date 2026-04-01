# UniKey-AI — Project Summary

> **A Python middleware library that intercepts BYOK (Bring Your Own Key) credentials from HTTP request headers, validates them, and injects a provider-native LangChain `BaseChatModel` into the developer's route handlers — so they write zero authentication or provider-routing boilerplate.**

---

## 1. The Problem Being Solved

Independent developers and small teams building Gen-AI applications face two structural problems:

**Problem 1 — Cost Liability.** When a developer deploys a chat app, RAG app, agentic AI app, or a docs-to-knowledge-graph pipeline, the LLM API key is theirs. Every request a user makes is billed to the developer. This is unsustainable for independent developers — they absorb all LLM costs for their users.

**Problem 2 — Provider Lock-in.** Developers currently hardcode which LLM provider and model their app uses. The user has no say, even if they already have an API key from a different provider. The developer decides the model; the developer pays the bill.

**The Solution — BYOK (Bring Your Own Key).** Let the end-user supply their own API key at request time. The user pays their own LLM costs directly. The developer's infrastructure becomes LLM-cost-free. The user picks the provider they already have access to.

The problem with implementing BYOK is that every developer building any Gen-AI app has to write the same boilerplate from scratch: extract headers, validate keys, route to the right provider SDK, catch auth errors, scrub keys from logs, and inject the configured LLM into their route logic. **UniKey-AI writes that boilerplate once, properly, and packages it for everyone.**

---

## 2. What UniKey-AI Is

UniKey-AI is a **Python middleware and Dependency Injection library** for web frameworks. It lives entirely at the **HTTP request boundary**. Its job begins when an incoming request arrives and ends the moment it hands the developer a fully configured, ready-to-use LLM object.

It does not touch application logic. It does not build chains, agents, or RAG pipelines. It does not persist API keys. It is purely **infrastructure glue** between a web request and an LLM object.

```text
[ Incoming HTTP Request ]
Headers: 
  x-ai-provider: openai / ollama 
  x-ai-key: sk-... / "dummy"
  x-ai-model: gpt-4o / llama3
  x-ai-base-url: (optional) http://localhost:11434
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ UniKey-AI Middleware Stack                                  │
│                                                             │
│  1. BYOKMiddleware (HTTP Exception Catcher)                 │
│     Catches Auth/Provider Errors occurring downstream       │
│     and safely converts them to standard HTTP 401/400s.     │
│                                                             │
│  2. FastApi Dependency: get_byok_llm()                      │
│     Extracts headers securely via dependency injection.     │
│                                                             │
│  3. LLM Factory: build_llm()                                │
│     Maps headers to a langchain-litellm ChatLiteLLM class.  │
│     Passes x-ai-base-url as api_base for local models.      │
└─────────────────────────────────────────────────────────────┘
          │
          │  Yields: BaseChatModel (ChatLiteLLM)
          ▼[ Developer's Route Handler (FastAPI) ]
  @app.post("/chat")
  def chat(llm = Depends(get_byok_llm())):
      return llm.invoke(prompt)  <-- Dev writes 1 line of code
          │
          ▼
[ LiteLLM Routing Layer ]
          │
  ┌───────┼────────┬─────────┬──────────────┐
  ▼       ▼        ▼         ▼              ▼
OpenAI  Anthropic Gemini   Groq    Local Models (Ollama/LM Studio)
                                   (Routed via x-ai-base-url)
```

---

## 3. The HTTP Contract — Request Headers

The frontend application or end-user sends the following standard HTTP headers with every Gen-AI request. 

| Header | Required? | Example Value | Description |
|---|---|---|---|
| `x-ai-provider` | **Yes** | `openai`, `anthropic`, `ollama` | The LLM provider name |
| `x-ai-key` | **Yes** | `sk-abc123...` | User's API key (can be 'dummy' for local) |
| `x-ai-model` | **Yes** | `gpt-4o`, `llama3` | The specific model string |
| `x-ai-base-url` | *Optional*| `http://localhost:11434` | **Crucial for local models** (Ollama, LM Studio, vLLM). Overrides provider defaults. |

These headers are the complete contract between the end-user and the library. The developer's backend never needs to manually read or store them — UniKey-AI intercepts, validates, uses, and discards them within the request lifecycle.

---

## 4. The Dependency Stack

### The Base — `langchain-litellm`

The core LLM object that UniKey-AI builds and injects is a `ChatLiteLLM` instance from the `langchain-litellm` package. This is the critical foundation choice.

`ChatLiteLLM` extends LangChain's `BaseChatModel`. This means the object UniKey-AI injects is a **first-class, native LangChain model** — not a wrapper, not a proxy, not a custom class. It is the same kind of object a developer would get from `ChatOpenAI` or `ChatAnthropic`, but provider-agnostic.

Internally, `ChatLiteLLM` uses LiteLLM to route API calls to 100+ providers. UniKey-AI uses this single class to support all providers through one dependency, with one interface.

```python
# What UniKey-AI's factory does internally — just this:
from langchain_litellm import ChatLiteLLM
from typing import Optional

def build_llm(
    provider: str, 
    api_key: str, 
    model: str, 
    api_base: Optional[str] = None
) -> ChatLiteLLM:
    """
    Constructs a LangChain BaseChatModel using LiteLLM.
    Handles standard providers and custom local base URLs.
    """
    kwargs = {
        "model": f"{provider}/{model}",
        "api_key": api_key,
        "custom_llm_provider": provider,
    }
    
    if api_base:
        kwargs["api_base"] = api_base
        
    return ChatLiteLLM(**kwargs)
```

### Full Dependency Chain

```
unikey-ai
    └── langchain-litellm          ← one package, all providers
            └── litellm            ← routes to 100+ provider APIs
            └── langchain-core     ← BaseChatModel, Runnable interface
```

UniKey-AI manages **zero provider-specific packages**. When LiteLLM adds a new provider, UniKey-AI supports it automatically with no code changes.

### What `ChatLiteLLM` (BaseChatModel) provides out of the box

Because the injected object is a proper `BaseChatModel`, the developer gets all standard LangChain capabilities without any additional setup:

| Capability | Available |
|---|---|
| `.invoke(prompt)` | ✅ |
| `.stream(prompt)` | ✅ |
| `.ainvoke(prompt)` (async) | ✅ |
| `.bind_tools(tools)` | ✅ |
| `.with_structured_output(schema)` | ✅ |
| Works in LCEL chains (`|` pipe operator) | ✅ |
| Works in LangGraph nodes | ✅ |
| Works as CrewAI agent LLM | ✅ |
| Retry logic (`max_retries`) | ✅ |
| Streaming support | ✅ |

---

## 5. The Developer Experience (DX)

The entire design goal is that a developer should be able to add BYOK support to any Gen-AI application in under **5 minutes** with under **5 lines of new code**.

### Without UniKey-AI (what every developer writes today):

```python
@app.post("/generate")
def generate(prompt: str, request: Request):
    provider = request.headers.get("x-ai-provider")
    api_key  = request.headers.get("x-ai-key")
    model    = request.headers.get("x-ai-model")

    if not api_key or not provider or not model:
        raise HTTPException(status_code=400, detail="Missing BYOK headers")

    try:
        llm = ChatLiteLLM(
            model=f"{provider}/{model}",
            api_key=api_key,
            custom_llm_provider=provider
        )
        response = llm.invoke(prompt)
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid API key provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"result": response.content}
```

This boilerplate is written and maintained separately in **every route**, in **every app**, by **every developer**.

### With UniKey-AI:

```python
from fastapi import FastAPI
from unikey_ai.fastapi import BYOKMiddleware, get_byok_llm

app = FastAPI()
app.add_middleware(BYOKMiddleware)

@app.post("/generate")
def generate(prompt: str, llm = Depends(get_byok_llm())):
    return {"result": llm.invoke(prompt).content}
```

The developer writes one middleware line and one dependency injection line. Every route that needs an LLM just declares `llm = Depends(get_byok_llm())` and receives a fully configured, provider-correct `BaseChatModel`.

---

## 6. What UniKey-AI Handles — Feature Set

### Header Extraction & Validation
Automatically parses the required and optional `x-ai-*` headers from the request. Validates presence and returns clean HTTP `400 Bad Request` errors if required headers are missing, *before* executing developer route logic.

### Universal LLM Initialization (`langchain-litellm`)
Translates headers into a fully configured `ChatLiteLLM` object. Because it extends LangChain's `BaseChatModel`, it natively supports `.invoke()`, `.stream()`, `.bind_tools()`, and works seamlessly with LangChain LCEL, LangGraph, and CrewAI.

### Local Model Support (Day One)
By accepting an optional `x-ai-base-url` header and passing it as `api_base` to LiteLLM, UniKey-AI inherently supports users running local models via Ollama, LM Studio, or any OpenAI-compatible local server.

### Graceful Error Handling via Middleware
When a developer calls `llm.invoke()`, the actual network request happens. If the user provided a bad API key, LiteLLM raises an authentication error deep in the route handler. `BYOKMiddleware` catches these provider-specific errors (`AuthenticationError`, `RateLimitError`) globally and translates them into clean HTTP `401 Unauthorized` or `429 Too Many Requests` responses. The developer writes *zero* `try/except` blocks.

---

## 7. LangChain Ecosystem Compatibility

Because the injected object is a native `BaseChatModel`, UniKey-AI is **automatically compatible** with every framework and tool in the LangChain ecosystem — with zero additional work from the developer or from UniKey-AI itself.

| Framework | How `llm` is used | Works? |
|---|---|---|
| **LangChain** (LCEL chains) | `prompt \| llm \| parser` | ✅ |
| **LangGraph** (agent nodes) | `llm.bind_tools(tools)` as node | ✅ |
| **CrewAI** (agent LLM) | `Agent(llm=llm, ...)` | ✅ |
| **LangChain RAG** | `RetrievalQA.from_llm(llm=llm)` | ✅ |
| **LangChain Agents** | `initialize_agent(llm=llm, tools=...)` | ✅ |

The developer's existing LangChain code changes by exactly **one thing**: instead of hardcoding `ChatOpenAI(api_key="...")`, they receive the `llm` via `Depends(get_byok_llm())`. Everything downstream — chains, graphs, agent loops, RAG pipelines — works without modification.

---

## 8. Package Structure

```text
unikey-ai/
├── unikey_ai/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── factory.py          # build_llm() mappings
│   │   └── exceptions.py       # UniKey custom exceptions
│   │
│   └── integrations/
│       ├── __init__.py
│       └── fastapi_utils.py    # BYOKMiddleware & get_byok_llm()
│
├── examples/
│   └── main.py                 # Example FastAPI app
├── requirements.txt
├── README.md
└── pyproject.toml
```

### `core/factory.py`
The only place in the library where `ChatLiteLLM` is instantiated. Receives `(provider, api_key, model, api_base)` as strings, returns a configured `ChatLiteLLM` object. Contains the provider string validation and normalization logic.

### `core/exceptions.py`
Custom exception classes that map to HTTP status codes: `MissingHeaderError → 400`, `InvalidKeyError → 401`, `ProviderNotSupportedError → 422`. These are caught by the middleware and converted to `HTTPException` before reaching the developer's code.

### `integrations/fastapi_utils.py`
Contains `BYOKMiddleware` (a `BaseHTTPMiddleware` subclass) and `get_byok_llm()` (a FastAPI dependency function that uses `Depends`). This is the file developers import from. Flask and Django equivalents will follow in later phases.

---

## 9. What UniKey-AI Is NOT

- It is **not** a LangChain replacement or extension. It uses LangChain objects as its output type.
- It is **not** a key vault or secrets manager. Keys are never stored, persisted, or cached.
- It is **not** a billing system. Token usage and costs are solely the responsibility of the end-user.
- It is **not** an agent framework. It does not build chains, graphs, or pipelines.
- It is **not** a proxy server. It runs inside the developer's existing FastAPI application.
- It is **not** tied to a specific LLM provider. It works with any provider LiteLLM supports.

---

## 10. Phased Rollout Plan

**Phase 1 — Core Infrastructure (Current Focus)**
- FastApi Middleware (`BYOKMiddleware`) for global error intercepting.
- Dependency Injection (`get_byok_llm()`) for dynamic header extraction.
- `ChatLiteLLM` factory for mapping provider + keys.
- **Support for local models via `x-ai-base-url` wrapper.**

**Phase 2 — Frontend Widget**
A drop-in React/JS component: an API key input modal with provider selection dropdown and live key validation feedback. Developers embed it in their frontend to collect the user's key before making requests.

**Phase 3 — Optional Proxy Mode**
A self-hostable lightweight proxy server (a FastAPI app using UniKey-AI itself) for non-Python backends that need BYOK header injection without embedding the library directly.
