"""Week 2, part 2 — the SAME assistant built twice: /chat vs /agent.

This is the hands-on answer to "Microsoft GenAI vs agentic AI" from your
side chat: agentic AI *uses* a GenAI model, it just adds tools + a loop.

    GenAI pattern    POST /chat:   retrieve -> build prompt -> model answers (one shot)
    Agentic pattern  POST /agent:  goal -> think -> call tool -> observe -> repeat

Run it next to the other two apps:

    uvicorn app.main_chat_agent:app --reload --port 8002

Then try the same question on all three and feel the difference:
    POST /chat  {"question": "What is RAG?"}
    POST /agent {"question": "What is RAG?"}
    POST /agent {"question": "What is RAG? Give me the exact details."}   # extra tool call

Fully offline — FakeLLM stands in for the model so you can see the
*wiring* (prompt assembly, tool loop, trace) without a key. Set
OPENAI_API_KEY and `pip install openai` to swap a real chat model in at
the two marked seams; the endpoints don't change.

.NET map for what's new here:
    Semantic Kernel plugin fn     ->  Tool.run() (name + description + schema)
    KernelFunction / FunctionTool ->  each Tool dict below; in Week 3 a tool
                                       becomes an MCP client call instead
    ChatCompletionAgent (Agent Framework) -> run_agent(): think, act, observe, repeat
    IChatClient                   ->  FakeLLM (swap for OpenAI/Azure OpenAI chat)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app import rag
from app.main_week2 import get_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Week1 AI Service — /chat vs /agent")


# --- Tools ---------------------------------------------------------------
# The shape every agent framework wants: name, description, JSON-ish
# arguments schema, and a callable. .NET: [KernelFunction] /
# Agent Framework FunctionTool. Week 3: run() calls an MCP server.

@dataclass
class Tool:
    name: str
    description: str
    arguments: dict[str, str]  # arg name -> type hint, kept simple on purpose
    run: Any  # callable(**args) -> str


def _search_docs(store: rag.InMemoryVectorStore, query: str, top_k: int = 3) -> str:
    results = store.search(query, top_k=top_k)
    if not results:
        return "no matches"
    return "\n".join(
        f"[{r.chunk.id}] ({r.chunk.source}, score={r.score:.3f}) {r.chunk.text}"
        for r in results
    )


def _read_chunk(store: rag.InMemoryVectorStore, chunk_id: str) -> str:
    for chunk in store._chunks:  # small demo store; a real one queries by id
        if chunk.id == chunk_id:
            return f"[{chunk.id}] {chunk.source}:\n{chunk.text}"
    return f"unknown chunk id: {chunk_id}"


def build_tools(store: rag.InMemoryVectorStore) -> list[Tool]:
    return [
        Tool(
            name="search_docs",
            description="Search the knowledge base for chunks relevant to a query.",
            arguments={"query": "str", "top_k": "int"},
            run=lambda query, top_k=3: _search_docs(store, query, top_k),
        ),
        Tool(
            name="read_chunk",
            description="Read the full text of one chunk by its id.",
            arguments={"chunk_id": "str"},
            run=lambda chunk_id: _read_chunk(store, chunk_id),
        ),
    ]


# --- Model seam ------------------------------------------------------------

class FakeLLM:
    """Deterministic stand-in for a chat model, so the wiring is visible
    offline. It does extractive answering from retrieved context — no real
    generation. .NET: swap this for IChatClient / Azure OpenAI chat."""

    def generate(self, prompt: str) -> str:
        # SEAM: replace with openai chat.completions.create(...) and the
        # rest of this module keeps working.
        lines = [ln for ln in prompt.splitlines() if ln.startswith("[")]
        cited = ", ".join(ln.split("]")[0][1:] for ln in lines[:3]) or "no sources"
        return f"(fake-llm) Answer assembled from [{cited}]. Set OPENAI_API_KEY for real generation."


# --- /chat: plain GenAI RAG -------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    source: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    prompt: str  # the assembled prompt — the part /ask (main_week2) leaves out
    sources: list[SourceChunk]


def build_prompt(question: str, results: list[rag.ScoredChunk]) -> str:
    context = "\n".join(
        f"[{r.chunk.id}] {r.chunk.source}: {r.chunk.text}" for r in results
    )
    return (
        "You answer using ONLY the context below. Cite chunk ids.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest, store: rag.InMemoryVectorStore = Depends(get_store)
) -> ChatResponse:
    # The whole GenAI pattern in one line: YOU orchestrate retrieve -> prompt -> model.
    results = store.search(req.question, top_k=req.top_k)
    prompt = build_prompt(req.question, results)
    answer = FakeLLM().generate(prompt)
    logger.info("chat: %r -> %d source(s)", req.question, len(results))
    return ChatResponse(
        answer=answer,
        prompt=prompt,
        sources=[SourceChunk(source=r.chunk.source, score=round(r.score, 4)) for r in results],
    )


# --- /agent: the same model, but it drives the tools -------------------------

class AgentStep(BaseModel):
    step: int
    thought: str
    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None


class AgentRequest(BaseModel):
    question: str = Field(min_length=1)
    max_steps: int = Field(default=4, ge=1, le=8)


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    tools_used: list[str]


def _wants_detail(question: str) -> bool:
    return any(
        cue in question.lower()
        for cue in ("detail", "exact", "quote", "full text", "more about", "verbatim")
    )


def run_agent(
    question: str, store: rag.InMemoryVectorStore, max_steps: int = 4
) -> tuple[str, list[AgentStep]]:
    """Think -> act -> observe loop. The policy below is deterministic so
    it's inspectable; a real agent lets the MODEL make these decisions
    (function calling / tool_choice). .NET: ChatCompletionAgent with
    FunctionTools does exactly this loop for you."""
    tools = {t.name: t for t in build_tools(store)}
    steps: list[AgentStep] = []
    retrieved: list[rag.ScoredChunk] | None = None

    for i in range(1, max_steps + 1):
        if retrieved is None:
            thought = "I don't have any context yet — search the docs first."
            tool_call, args = "search_docs", {"query": question, "top_k": 3}
        elif _wants_detail(question) and not any(s.tool == "read_chunk" for s in steps):
            best = retrieved[0].chunk.id
            thought = f"The question asks for detail — read the full text of {best}."
            tool_call, args = "read_chunk", {"chunk_id": best}
        else:
            steps.append(AgentStep(step=i, thought="I have what I need — synthesize the answer."))
            break

        observation = str(tools[tool_call].run(**args))
        if tool_call == "search_docs":
            retrieved = store.search(question, top_k=3)
        steps.append(
            AgentStep(step=i, thought=thought, tool=tool_call, arguments=args, observation=observation)
        )

    context = "\n".join(s.observation for s in steps if s.observation) or "no context"
    answer = FakeLLM().generate(f"QUESTION: {question}\nCONTEXT:\n{context}\nANSWER:")
    return answer, steps


@app.post("/agent", response_model=AgentResponse)
def agent(
    req: AgentRequest, store: rag.InMemoryVectorStore = Depends(get_store)
) -> AgentResponse:
    answer, steps = run_agent(req.question, store, max_steps=req.max_steps)
    tools_used = [s.tool for s in steps if s.tool]
    logger.info("agent: %r -> %d step(s), tools=%s", req.question, len(steps), tools_used)
    return AgentResponse(answer=answer, steps=steps, tools_used=tools_used)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
