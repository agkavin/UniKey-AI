# UniKey-AI

> **BYOK (Bring Your Own Key) middleware for FastAPI.** Intercepts `x-ai-*` request headers, validates them, and injects a provider-native LangChain `BaseChatModel` into your route handlers — zero authentication boilerplate.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Quickstart

```python
from fastapi import FastAPI, Depends
from unikey_ai import BYOKMiddleware, get_byok_llm

app = FastAPI()
app.add_middleware(BYOKMiddleware)  # Global error handler

@app.post("/chat")
async def chat(prompt: str, llm=Depends(get_byok_llm())):
    response = await llm.ainvoke(prompt)
    return {"result": response.content}
```

That's it. No API key management, no provider routing, no try/except blocks.

---

## HTTP Header Contract

Every request must include these headers:

| Header | Required | Example |
|--------|----------|---------|
| `x-ai-provider` | ✅ | `openai`, `anthropic`, `groq`, `ollama` |
| `x-ai-key` | ✅ | `sk-abc123...` (use `dummy` for local models) |
| `x-ai-model` | ✅ | `gpt-4o-mini`, `llama3`, `claude-3-5-sonnet-20241022` |
| `x-ai-base-url` | Optional | `http://localhost:11434` (required for local models) |

### Supported Providers (Phase 1)

`openai` · `anthropic` · `gemini` · `groq` · `cohere` · `mistral` · `ollama` · `openai-compatible`

---

## What the Injected LLM Supports

The injected object is a first-class LangChain `BaseChatModel` — it works everywhere:

```python
# Direct invocation
response = await llm.ainvoke("Hello!")

# Streaming
async for chunk in llm.astream("Hello!"):
    print(chunk.content, end="")

# LCEL chain
chain = prompt | llm | StrOutputParser()

# LangGraph node
llm_with_tools = llm.bind_tools(tools)

# Structured output
llm.with_structured_output(MySchema)
```

---

## Error Handling

`BYOKMiddleware` catches everything automatically:

| Situation | HTTP Response |
|-----------|---------------|
| Missing required header | `400 Bad Request` |
| Unknown provider | `422 Unprocessable Entity` |
| Invalid / rejected API key | `401 Unauthorized` |
| Provider rate limit hit | `429 Too Many Requests` |
| Unexpected server error | `500 Internal Server Error` |

---

## Local Models (Ollama, LM Studio, vLLM)

```bash
# Start Ollama
ollama serve

# Request to your app
curl -X POST http://localhost:8000/chat \
  -H "x-ai-provider: ollama" \
  -H "x-ai-key: dummy" \
  -H "x-ai-model: llama3" \
  -H "x-ai-base-url: http://localhost:11434" \
  -d '{"prompt": "Hello!"}'
```

---

## Advanced: Enforce Base URL for Specific Providers

```python
@app.post("/local")
async def local_chat(llm=Depends(get_byok_llm(require_base_url_for=["ollama"]))):
    # Returns 400 if x-ai-base-url is missing when provider is ollama
    return {"result": (await llm.ainvoke("Hi!")).content}
```

---

## Run the Example App

```bash
uvicorn examples.main:app --reload
# Open http://localhost:8000/docs
```

## Run Tests

```bash
pytest tests/ -v
```

---

## Package Structure

```
unikey_ai/
├── __init__.py                    # Public API
├── core/
│   ├── exceptions.py              # Error hierarchy (→ HTTP codes)
│   └── factory.py                 # build_llm() — the only LLM constructor
└── integrations/
    └── fastapi_utils.py           # BYOKMiddleware + get_byok_llm()
examples/
└── main.py                        # Working FastAPI demo
tests/
├── test_factory.py                # Unit tests (no network)
└── test_middleware.py             # Integration tests (monkeypatched)
```
