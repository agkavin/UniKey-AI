"""
UniKey-AI — Example FastAPI Application
========================================

Demonstrates all Phase 1 features:
  - BYOKMiddleware for global error handling
  - get_byok_llm() dependency for LLM injection
  - Sync and async invocation
  - LCEL chain usage
  - Streaming endpoint
  - Local model (Ollama) usage via x-ai-base-url

Run with:
    uvicorn examples.main:app --reload

Test with curl:
    # Cloud model (replace with your real key)
    curl -X POST "http://localhost:8000/chat" \\
      -H "Content-Type: application/json" \\
      -H "x-ai-provider: openai" \\
      -H "x-ai-key: sk-YOUR_KEY" \\
      -H "x-ai-model: gpt-4o-mini" \\
      -d '{"prompt": "What is the capital of France?"}'

    # Local Ollama model (no real key needed)
    curl -X POST "http://localhost:8000/chat" \\
      -H "Content-Type: application/json" \\
      -H "x-ai-provider: ollama" \\
      -H "x-ai-key: dummy" \\
      -H "x-ai-model: llama3" \\
      -H "x-ai-base-url: http://localhost:11434" \\
      -d '{"prompt": "What is the capital of France?"}'

    # Test 400 error (missing header)
    curl -X POST "http://localhost:8000/chat" \\
      -H "Content-Type: application/json" \\
      -H "x-ai-provider: openai" \\
      -d '{"prompt": "Hello"}'

    # Test 401 error (bad key)
    curl -X POST "http://localhost:8000/chat" \\
      -H "Content-Type: application/json" \\
      -H "x-ai-provider: openai" \\
      -H "x-ai-key: sk-INVALID" \\
      -H "x-ai-model: gpt-4o-mini" \\
      -d '{"prompt": "Hello"}'
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from unikey_ai import BYOKMiddleware, get_byok_llm

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UniKey-AI Demo",
    description="Example app demonstrating BYOK middleware and dependency injection.",
    version="0.1.0",
)

# Register the middleware ONCE — it wraps every route automatically.
# All LiteLLM and UniKey-AI errors are caught here and returned as clean JSON.
app.add_middleware(BYOKMiddleware)


# ─── Request / Response models ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    result: str
    provider: str | None = None
    model: str | None = None


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    """Health check and usage instructions."""
    return {
        "service": "UniKey-AI Demo",
        "status": "running",
        "endpoints": {
            "POST /chat": "Basic chat — returns the model's reply as a string.",
            "POST /chat/chain": "LCEL chain demo — uses prompt template + LLM + parser.",
            "POST /chat/stream": "Streaming response — returns tokens as they arrive.",
        },
        "required_headers": {
            "x-ai-provider": "openai | anthropic | gemini | groq | cohere | mistral | ollama | openai-compatible",
            "x-ai-key": "Your API key (use 'dummy' for local models)",
            "x-ai-model": "Model name e.g. gpt-4o-mini, llama3, claude-3-5-sonnet-20241022",
            "x-ai-base-url": "(Optional) Base URL for local models e.g. http://localhost:11434",
        },
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    llm=Depends(get_byok_llm()),
):
    """
    Basic async chat endpoint.

    The `llm` parameter is a fully configured BaseChatModel injected by
    get_byok_llm(). The developer writes exactly one line to call the model.
    """
    response = await llm.ainvoke([HumanMessage(content=body.prompt)])
    return ChatResponse(
        result=response.content,
        provider=request.headers.get("x-ai-provider"),
        model=request.headers.get("x-ai-model"),
    )


@app.post("/chat/chain")
async def chat_with_chain(
    body: ChatRequest,
    llm=Depends(get_byok_llm()),
):
    """
    LCEL chain demo.

    Demonstrates that the injected BaseChatModel works directly in a LangChain
    Expression Language (LCEL) pipe chain without any modification.
    The llm object is a real BaseChatModel — it supports the | pipe operator natively.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a concise, helpful assistant. Keep answers under 3 sentences."),
            ("human", "{user_input}"),
        ]
    )

    # Standard LCEL chain: prompt | llm | output_parser
    chain = prompt_template | llm | StrOutputParser()
    result = await chain.ainvoke({"user_input": body.prompt})

    return {"result": result}


@app.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    llm=Depends(get_byok_llm()),
):
    """
    Streaming chat endpoint.

    Streams tokens back as they are generated using Server-Sent Events (SSE).
    The injected BaseChatModel supports .astream() natively — no extra setup.
    """

    async def token_generator() -> AsyncGenerator[str, None]:
        async for chunk in llm.astream([HumanMessage(content=body.prompt)]):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(token_generator(), media_type="text/event-stream")
