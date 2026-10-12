# Week 1 — FastAPI starter (for a .NET engineer)

Built for Simeon's Python-for-AI-engineering plan (goal `goal_285a0b133eae`).
This is the "small FastAPI build" starting option — the alternative is to port
one of your existing .NET endpoints instead. Everything below maps a concept
you already know in ASP.NET Core to its Python/FastAPI equivalent.

## Python version — pin to 3.13, skip 3.15 for now

Python 3.15.0rc3 is out and the 3.15 final release is scheduled for
**2026-10-09**. For AI work, stay on **Python 3.13**: LangChain 1.0 requires
3.10+, and current LangChain/LangGraph courses and examples target 3.12/3.13.
New CPython releases routinely lag on prebuilt wheels for heavier packages,
so learn on 3.13 and upgrade once the ecosystem catches up. `.python-version`
in this folder already pins 3.13.

## Setup (VS Code)

```bash
cd week1-fastapi-starter
py -3.13 -m venv .venv        # or: uv venv --python 3.13
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs — FastAPI's built-in Swagger UI,
equivalent to the Swagger page you get from Swashbuckle / `AddEndpointsApiExplorer`.

## .NET → FastAPI map (the 80% you need this week)

| ASP.NET Core | FastAPI / Python |
| --- | --- |
| `WebApplication.CreateBuilder()` | `app = FastAPI()` |
| `app.MapGet("/health", ...)` | `@app.get("/health")` |
| `[FromBody]` DTO record | A `pydantic.BaseModel` class |
| Data annotations (`[Required]`) | Pydantic field types/validators — validation is automatic |
| `AddSingleton<T>()` / DI | Module-level object, or `Depends()` for per-request deps |
| `appsettings.json` + `IOptions<T>` | `.env` + `pydantic-settings` (`Settings` class) |
| `ILogger<T>` | `logging.getLogger(__name__)` |
| NuGet / `.csproj` | `pip` + `requirements.txt` (or `uv` + `pyproject.toml`) |
| `dotnet run` | `uvicorn app.main:app --reload` |
| xUnit / `WebApplicationFactory` | `pytest` + `fastapi.testclient.TestClient` |

## What's in `app/main.py`

- `GET /health` — liveness, like a minimal-API health endpoint.
- `POST /ask` — a RAG-shaped request/response stub (question in, answer +
  sources out). The retrieval step is a TODO where your existing RAG
  experience plugs in during Week 2 — embeddings/vector DB wiring is
  deliberately left as the exercise, since you already know the concepts.

## Suggested first session (30–45 min)

1. Run the setup above, open `/docs`, call both endpoints.
2. Add one field to `AskRequest` (e.g. `top_k: int = 3`) and watch Pydantic
   validate it — compare with how a C# DTO + ModelState behaves.
3. Break it on purpose: send `{"question": 42}` and read the 422 response.
   That's your `[ApiController]` automatic validation, for free.

## Week 2 — RAG wired, then `/chat` vs `/agent`

Week 2 is already scaffolded here; run each app on its own port.

### `app/main_week2.py` — the Week 1 stub, with real retrieval (port 8001)

```bash
uvicorn app.main_week2:app --reload --port 8001
```

`POST /ask {"question": "What is RAG?", "top_k": 2}` now embeds the
question, searches an in-memory store seeded with four chunks
(`app/rag.py`), and returns ranked sources with cosine scores. Runs fully
offline: `rag.get_embedding_provider()` falls back to a deterministic fake
provider; set `OPENAI_API_KEY` (and `pip install openai`) and the same code
path uses `text-embedding-3-small` — nothing at the call site changes.

The `.NET` map is in the docstrings: `EmbeddingProvider` (a `Protocol`,
the structural-interface equivalent of `IEmbeddingGenerator`),
`InMemoryVectorStore`, and `get_store()` as the `Depends()` seam. The
lifespan handler seeds once at boot (your `IHostedService`), and the store
also builds lazily so plain `TestClient` tests work without the lifespan
context manager.

### `app/main_chat_agent.py` — GenAI vs agentic AI, same assistant twice (port 8002)

```bash
uvicorn app.main_chat_agent:app --reload --port 8002
```

- `POST /chat` — plain **GenAI RAG**: retrieve → assemble the prompt →
  model answers. One shot; you orchestrate. The response includes the
  assembled `prompt` so you can see prompt construction — the step `/ask`
  in `main_week2` leaves out.
- `POST /agent` — **agentic**: the same retrieval wrapped as tools
  (`search_docs`, `read_chunk`), driven by a think → act → observe loop.
  The response includes the full `steps` trace. Try a plain question (one
  tool call) versus `"What is RAG? Give me the exact details."` (two) —
  that's the core agent property: the *loop* decides whether a second
  tool call is needed, not your code.

Both run offline via `FakeLLM`; the two model seams are marked for a real
chat model later. Tool shape (name + description + args + `run`) mirrors
`KernelFunction`/`FunctionTool` in .NET Agent Framework — and is exactly
what an MCP tool call looks like, so Week 3's MCP wiring slots in at the
same seam.

### `agent_loop.py` — the same agent, no FastAPI, real model when keyed

```bash
python agent_loop.py "What is RAG? Give me the exact details."   # add --offline to force the fake policy
```

One file, no framework: the same `search_docs` + `read_chunk` tools, but the
loop is driven by a real model via OpenAI function calling — the model
chooses the tools and when to stop, instead of the deterministic policy in
`/agent`. Needs `pip install openai` + `OPENAI_API_KEY`; without a key it
falls back to the `/agent` policy so the mechanics still run.

### `agent_memory.py` — the same agent, with multi-turn memory

```bash
python agent_memory.py              # scripted 2-turn demo, memory ON
python agent_memory.py --no-memory  # same demo, memory OFF (watch turn 2)
python agent_memory.py --interactive
```

`agent_loop.py` is stateless, so a follow-up like *"give me the exact
details"* has no referent. This adds the article's "Adding memory" piece: a
`Conversation` class holds the message list across turns (the .NET
`AgentThread`/`ChatHistory` equivalent), with a trim policy bounding the
context window and `save()`/`load()` for persistence. Run with `--no-memory`
to see the failure mode — turn 2 searches its own dangling text and reads an
arbitrary chunk. See [WEEK2_MEMORY.md](WEEK2_MEMORY.md) for the exercises
(summarise-instead-of-truncate trim, session memory in `/agent`, real-model
flip).
