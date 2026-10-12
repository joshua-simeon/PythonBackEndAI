"""Week 2 wired version — same starter, with real retrieval behind /ask.

Run it side-by-side with the Week 1 stub:

    uvicorn app.main_week2:app --reload --port 8001

Then compare:
    POST /ask {"question": "What is RAG?", "top_k": 2}

Week 1 (app/main.py) returns a stub source. This file embeds the question,
searches an in-memory store seeded with four chunks (app/rag.py), and returns
ranked sources with cosine scores. It runs fully offline — no API key needed
— because rag.get_embedding_provider() falls back to a deterministic fake
provider. Set OPENAI_API_KEY (and `pip install openai`) and the same code
path uses real embeddings; nothing at the call site changes.

.NET map for what's new here:
    AddSingleton<IVectorStore>()     ->  module-level `store` + Depends(get_store)
    IOptions<T> / appsettings        ->  env var checked once in the factory
    ILogger<T>                       ->  logging.getLogger(__name__) (same as Week 1)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from app import rag

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ASP.NET: IHostedService / startup. Seed once at boot, not per request.
    store = rag.build_store()
    app.state.store = store
    logger.info("seeded %d chunks", len(store))
    yield


app = FastAPI(title="Week1 AI Service — Week 2 RAG wired", lifespan=lifespan)


def get_store() -> rag.InMemoryVectorStore:
    """Depends() target — the per-request DI seam. .NET: [FromServices].

    Builds lazily if lifespan didn't run (e.g. a bare TestClient without
    the context manager), so the app is testable the same way you'd test
    a minimal API with WebApplicationFactory: no special harness needed.
    """
    store = getattr(app.state, "store", None)
    if store is None:
        store = rag.build_store()
        app.state.store = store
    return store


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, store: rag.InMemoryVectorStore = Depends(get_store)) -> AskResponse:
    results = store.search(req.question, top_k=req.top_k)
    sources = [SourceChunk(source=r.chunk.source, score=round(r.score, 4)) for r in results]
    if results:
        top = results[0].chunk
        answer = (
            f"Based on {top.source}: {top.text} "
            f"(+{len(results) - 1} more source(s) — prompt assembly is the next step.)"
        )
    else:
        answer = "No sources in the store yet."
    logger.info("ask: %r -> %d source(s)", req.question, len(sources))
    return AskResponse(answer=answer, sources=sources)
