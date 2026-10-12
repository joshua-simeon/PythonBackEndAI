"""Week 1 FastAPI starter — written for a .NET engineer.

Each block notes the ASP.NET Core equivalent so you can read this
through the lens of what you already know.
"""

# ASP.NET: using directives. Python: imports at module top.
import logging
from pydantic import BaseModel, Field
from fastapi import FastAPI

# ASP.NET: WebApplication.CreateBuilder() + builder.Build()
app = FastAPI(title="Week1 AI Service")

logger = logging.getLogger(__name__)  # ASP.NET: ILogger<T>


# ASP.NET: a DTO record, e.g. `public record AskRequest(string Question);`
# Pydantic validates automatically — the [Required]/DataAnnotations equivalent
# is the type annotation and Field(...) constraints.
class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = 3  # how many chunks to retrieve (Week 2 wires this up)


class SourceChunk(BaseModel):
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


# ASP.NET: app.MapGet("/health", () => Results.Ok(...))
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ASP.NET: app.MapPost("/ask", (AskRequest req) => { ... })
# The type hints do the model binding AND the OpenAPI schema generation.
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    # TODO (Week 2): replace this stub with your RAG pipeline —
    # embed req.question, query your vector DB, build the prompt.
    # You already know this flow from .NET; only the syntax is new.
    logger.info("ask: %s (top_k=%d)", req.question, req.top_k)
    return AskResponse(
        answer=f"(stub) You asked: {req.question}",
        sources=[SourceChunk(source="todo-vector-db", score=0.0)],
    )
