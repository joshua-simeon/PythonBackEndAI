# Lesson 1: Document API

A small starting point for a future document ingestion / RAG service.

## Run in PowerShell

Open a terminal in this folder, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.
Use Ctrl+C to stop the server. Reload is for local development.

## Try the API

1. Call GET /health.
2. Call POST /documents with the body below.
3. Copy the returned id into GET /documents/{document_id}.
4. Try an empty title: validation returns HTTP 422.
5. Try a valid UUID that does not exist: the API returns HTTP 404.

```json
{
  "title": "Company policy",
  "content": "Employees can request reimbursement for approved training."
}
```

## ASP.NET Core connections

| ASP.NET Core | This project |
| --- | --- |
| Application setup / Program.cs | app = FastAPI(...) |
| Controller / route group | routers/documents.py with APIRouter |
| MapGet / MapPost | @router.get / @router.post |
| Request DTO + validation attributes | Pydantic BaseModel + Field |
| Typed response contract | response_model |
| NotFound() | HTTPException(status_code=404, ...) |
| Guid route parameter | document_id: UUID |

Pydantic models describe and validate data; they are not ORM entities.
FastAPI infers the JSON body from the Pydantic parameter type and the route
parameter from its name. The UUID annotation validates and converts the route value.

These handlers use def because they do no asynchronous I/O. When adding an
async HTTP or database client, use async def and await. Declaring async def
alone does not make blocking operations asynchronous.

Storage is intentionally temporary. It resets on restart, each worker has its
own copy, and this sample is not a production persistence layer.

## Next lessons

1. Replace in-memory storage with database persistence.
2. Add database persistence and meaningful integration tests.
3. Add configuration, authentication, and logging.
4. Add AI document ingestion, retrieval, and evaluation.

## Documents router

`main.py` configures the application and registers the documents router with
`app.include_router(documents_router)`. `routers/documents.py` groups document
endpoints. DTOs live in `schemas.py`; storage operations live in `document_service.py`. Its `/documents` prefix applies to every
route, and its `Documents` tag groups them in Swagger UI. This is the FastAPI
equivalent of grouping actions in an ASP.NET Core controller; a controller class
is not required. The run command and endpoint URLs remain the same.

## Document service and dependency injection

`DocumentService` owns create/list/get operations, like an ASP.NET Core service
called by controller actions. `schemas.py` contains the unchanged Pydantic DTOs,
so the service does not import the HTTP router. The router retains HTTP concerns,
including the existing 404 response and response models. The health router is unchanged.

`Annotated[DocumentService, Depends(get_document_service)]` requests injection
into an endpoint parameter, comparable to a minimal API service parameter or
`[FromServices]`. FastAPI calls the provider; there is no central DI registration.
The provider returns a module-level service instance, analogous to
`AddSingleton<DocumentService>()` for this single-app process.

Service and dictionary lifetimes are both per worker process. Separate requests
share them. Restarting or development reloading clears them; multiple workers
have independent stores. Creating a new service inside the provider would create
new storage each request and lose earlier documents. FastAPI's default dependency
cache only reuses a provider result within one request; the module-level instance
provides the cross-request lifetime here. A lock protects dictionary access because
synchronous endpoints can run concurrently in worker threads.

Tests can replace the provider through `app.dependency_overrides`, comparable to
replacing a service registration in an ASP.NET Core integration-test host.
