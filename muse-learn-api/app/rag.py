"""Week 2 preview — RAG wiring for a .NET engineer.

You already know this pipeline from .NET (embed -> vector search -> prompt).
This file shows the idiomatic Python shape of the same pieces:

    .NET                                Python here
    ----------------------------------  --------------------------------
    IEmbeddingGenerator<T>              EmbeddingProvider (Protocol)
    OpenAI / Azure OpenAI client        OpenAIEmbeddingProvider
    InMemoryVectorStore / Qdrant etc.   InMemoryVectorStore
    VectorStoreRecord / chunk record    Chunk (dataclass)
    DI via AddSingleton<>               module singleton + Depends()

No API key needed to run it: FakeEmbeddingProvider is deterministic and
offline, so you can see retrieval working end-to-end today. Swap in
OpenAIEmbeddingProvider (set OPENAI_API_KEY) when you want real embeddings
— the call site in main.py doesn't change.
"""

from __future__ import annotations
import logging
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

logger = logging.getLogger(__name__)  # ASP.NET: ILogger<T>
# --- Embeddings ------------------------------------------------------------

class EmbeddingProvider(Protocol):
    """The Python equivalent of .NET's IEmbeddingGenerator.

    A Protocol is a structural interface: any class with an `embed`
    method matches, no explicit `: IEmbeddingGenerator` inheritance needed.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class FakeEmbeddingProvider:
    """Deterministic, offline embeddings for local dev / tests.

    A hashing vectorizer: each token stamps a few positions in a wide
    sparse vector, then the result is L2-normalised. Texts sharing words
    land close together (lexical overlap, no synonym knowledge), so
    wiring, ranking, and top_k are all testable without a key or a
    network call. Semantically shallow, but the *shape* is identical to
    a real provider. Direct analogue of the fake/mock embedding
    generator you'd register in a .NET test host.
    """

    def __init__(self, dimensions: int = 1024, stamps_per_token: int = 8) -> None:
        self.dimensions = dimensions
        self.stamps_per_token = stamps_per_token

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        # Tiny stopword list so "what is RAG?" is driven by "rag", not by
        # the stopwords it happens to share with unrelated chunks.
        stopwords = {
            "a", "an", "and", "are", "do", "does", "for", "how", "i", "in",
            "is", "it", "of", "or", "should", "the", "to", "what", "which",
        }
        tokens = [
            t
            for t in "".join(c if c.isalnum() else " " for c in text.lower()).split()
            if t not in stopwords
        ]
        vec = [0.0] * self.dimensions
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for k in range(self.stamps_per_token):
                # Two digest bytes per stamp: one picks the position,
                # one picks the sign, so collisions mostly cancel out.
                pos = int.from_bytes(digest[2 * k : 2 * k + 2], "little") % self.dimensions
                vec[pos] += 1.0 if digest[16 + k] % 2 == 0 else -1.0
        return _normalise(vec)


class OpenAIEmbeddingProvider:
    """Real embeddings via the OpenAI API (pip install openai).

    Mirrors your .NET OpenAI client setup: key from environment,
    model chosen once at construction (IOptions pattern -> constructor arg).
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        # Imported lazily so the module loads fine without the package
        # or a key when you're still on the fake provider.
        from openai import OpenAI  # type: ignore[import-not-found]

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. Using FakeEmbeddingProvider for offline dev."
            )
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use FakeEmbeddingProvider for "
                "offline dev, or set the key to use real embeddings."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in response.data]


def get_embedding_provider() -> EmbeddingProvider:
    """Factory — the Depends() target. .NET equivalent: the DI registration.

    Returns the OpenAI provider when a key is present, otherwise the fake.
    Register this once and inject it; don't `new` a provider per request.
    """
    if os.environ.get("OPENAI_API_KEY"):
        try:
            logger.info("OPENAI_API_KEY set — using OpenAIEmbeddingProvider")
            return OpenAIEmbeddingProvider()
        except (RuntimeError, ImportError):
            pass
    logger.info("OPENAI_API_KEY not set or openai package missing — using FakeEmbeddingProvider")
    return FakeEmbeddingProvider()


# --- Vector store ------------------------------------------------------------

@dataclass
class Chunk:
    """A stored document chunk. .NET: your vector-store record type."""

    id: str
    text: str
    source: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # Both sides are normalised at write/query time, so dot == cosine.
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore:
    """Smallest useful vector store. .NET: InMemoryVectorStore from
    Microsoft.Extensions.VectorData / Semantic Kernel samples.

    Swap this class for Qdrant/pgvector/Chroma later — the search
    signature (query embedding in, scored chunks out) stays the same.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider
        self._chunks: list[Chunk] = []

    def upsert(self, chunks: list[Chunk]) -> None:
        missing = [c for c in chunks if not c.embedding]
        if missing:
            embeddings = self._provider.embed([c.text for c in missing])
            for chunk, emb in zip(missing, embeddings):
                chunk.embedding = _normalise(emb)
        self._chunks.extend(chunks)

    def search(self, query: str, top_k: int = 3) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        [query_embedding] = self._provider.embed([query])
        query_embedding = _normalise(query_embedding)
        scored = [
            ScoredChunk(chunk=c, score=_cosine(query_embedding, c.embedding))
            for c in self._chunks
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)


# --- Seed data (so /ask returns something real immediately) ------------------

SEED_CHUNKS: list[Chunk] = [    Chunk(
        id="rag-1",
        source="notes/rag-overview.md",
        text=(
            "Retrieval-augmented generation (RAG) grounds an LLM response in "
            "retrieved documents: embed the question, search a vector store "
            "for the nearest chunks, then build a prompt from those chunks."
        ),
    ),
    Chunk(
        id="emb-1",
        source="notes/embeddings.md",
        text=(
            "Embeddings work by mapping text to a dense vector. Texts with "
            "similar meaning land close together, so cosine similarity over "
            "embedding vectors is a practical proxy for semantic similarity."
        ),
    ),
    Chunk(
        id="mcp-1",
        source="notes/mcp.md",
        text=(
            "The Model Context Protocol (MCP) is a standard way for AI apps to "
            "connect to external tools and data sources through a common "
            "client/server interface."
        ),
    ),
    Chunk(
        id="vec-1",
        source="notes/vector-db.md",
        text=(
            "A vector database indexes embeddings for fast approximate "
            "nearest-neighbour search. In .NET you might use Qdrant or "
            "pgvector; Python options include Chroma, FAISS, and pgvector."
        ),
    ),
]


def build_store(chunks: Sequence[Chunk] | None = None) -> InMemoryVectorStore:
    """Seeded store builder — the shared startup step for every Week 2 app.

    .NET: the code you'd put in an IHostedService.StartAsync or just after
    WebApplicationBuilder.Build(). main_week2 and main_chat_agent both call
    this so the two apps always start from the same data.
    """
    store = InMemoryVectorStore(get_embedding_provider())
    store.upsert(list(chunks) if chunks is not None else list(SEED_CHUNKS))
    return store
