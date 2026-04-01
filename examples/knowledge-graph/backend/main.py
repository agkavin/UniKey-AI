"""
Knowledge Graph Extractor — FastAPI Backend
============================================

Uses UniKey-AI BYOK middleware to extract entities and relationships from any
document and return a structured graph (nodes + edges) for frontend visualization.

Run:
    uvicorn main:app --reload --port 8001

Test:
    curl -X POST http://localhost:8001/extract \\
      -H "Content-Type: application/json" \\
      -H "x-ai-provider: groq" \\
      -H "x-ai-key: your-key" \\
      -H "x-ai-model: llama-3.3-70b-versatile" \\
      -d '{"text": "Elon Musk founded SpaceX in 2002. He is also CEO of Tesla."}'
"""

from __future__ import annotations

import json
import re
import logging
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from unikey_ai import BYOKMiddleware, get_byok_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="UniKey-AI Knowledge Graph Extractor",
    description="BYOK-powered entity and relationship extraction from text.",
    version="0.1.0",
)

app.add_middleware(BYOKMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175", "http://127.0.0.1:5175"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Graph Schema ─────────────────────────────────────────────────────────────


class Node(BaseModel):
    id: str
    label: str
    type: str  # Person | Organization | Location | Concept | Event | Other


class Edge(BaseModel):
    source: str
    target: str
    label: str


class KnowledgeGraph(BaseModel):
    nodes: list[Node]
    edges: list[Edge]


# ─── Request / Response ───────────────────────────────────────────────────────


class ExtractRequest(BaseModel):
    text: str


class ExtractResponse(BaseModel):
    graph: KnowledgeGraph
    node_count: int
    edge_count: int


# ─── Extraction Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledge graph extraction engine.

Given a piece of text, extract ALL named entities and the relationships between them.

Return ONLY a valid JSON object with this exact schema:
{
  "nodes": [
    {"id": "1", "label": "Entity Name", "type": "EntityType"}
  ],
  "edges": [
    {"source": "1", "target": "2", "label": "relationship description"}
  ]
}

Rules:
- Node IDs must be unique strings (use sequential numbers as strings: "1", "2", "3"…)
- Node types MUST be exactly one of: Person, Organization, Location, Concept, Event, Other
- Edge labels should be concise verb phrases (e.g. "founded", "CEO of", "located in")
- Only add edges between nodes that exist in the nodes list
- Extract at minimum 3 nodes and 2 edges if the text contains enough information
- Return ONLY the JSON object, no markdown, no explanation
"""


def _parse_json_fallback(text: str) -> KnowledgeGraph:
    """Extract JSON from LLM response that may contain extra text."""
    # Try to find a JSON object in the response
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return KnowledgeGraph.model_validate_json(match.group())


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    return {
        "service": "UniKey-AI Knowledge Graph Extractor",
        "status": "running",
        "endpoints": {
            "POST /extract": "Extract a knowledge graph from text using BYOK LLM",
        },
        "required_headers": {
            "x-ai-provider": "openai | anthropic | groq | gemini | mistral | cohere | ollama | openai-compatible",
            "x-ai-key": "Your API key for the chosen provider",
            "x-ai-model": "Model identifier, e.g. llama-3.3-70b-versatile",
            "x-ai-base-url": "(Optional) For local models, e.g. http://localhost:11434",
        },
    }


@app.post("/extract", response_model=ExtractResponse)
async def extract_graph(
    body: ExtractRequest,
    llm=Depends(get_byok_llm()),
):
    """
    Extract a knowledge graph from the provided text.

    The LLM is injected via UniKey-AI from the request headers.
    Tries structured output first; falls back to JSON parsing from raw text
    for models that don't support tool/function calling (e.g. Ollama).
    """
    if not body.text.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Extract the knowledge graph from this text:\n\n{body.text}"),
    ]

    graph: Optional[KnowledgeGraph] = None

    # Attempt 1: structured output (works with OpenAI, Groq, Anthropic, Gemini)
    try:
        structured_llm = llm.with_structured_output(KnowledgeGraph)
        graph = await structured_llm.ainvoke(messages)
        logger.info(f"Structured output succeeded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    except Exception as e:
        logger.warning(f"Structured output failed ({e}), falling back to JSON parsing")

    # Attempt 2: raw text + JSON extraction fallback (for Ollama, etc.)
    if graph is None:
        raw_response = await llm.ainvoke(messages)
        graph = _parse_json_fallback(raw_response.content)
        logger.info(f"Fallback JSON parsing succeeded: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    return ExtractResponse(
        graph=graph,
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
    )
