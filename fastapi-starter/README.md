# Document API: FastAPI and PostgreSQL

Run commands below from `fastapi-starter` with the root virtual environment activated.

## Configuration and run

This workspace uses the existing `my-postgres` PostgreSQL 16 container on localhost:5432
and a dedicated `fastapi_documents` database. No Docker Compose file is needed.
Start the existing container if stopped: `docker start my-postgres`.

For a new checkout, copy `.env.example` to `.env` and supply your local database
credentials. Create the database first. `.env` is ignored by Git; never commit it.
Environment variables override `.env`. URL-encode special characters in credentials.
The `postgresql+psycopg` scheme explicitly selects psycopg 3, not psycopg2.

```powershell
python -m pip install -r requirements.txt
python -m alembic upgrade head
python -m uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs. POST /documents accepts:

```json
{"title": "Company policy", "content": "Training reimbursement policy"}
```

Use the returned id with GET /documents/{document_id}; GET /documents lists documents.
GET /health returns {"status": "ok"}; it is a liveness endpoint, not a database probe.
Invalid bodies/UUIDs return 422. An absent valid UUID returns 404 with
{"detail": "Document not found"}. Response fields remain id, title, content.
Listing has no defined ordering.

## EF Core comparisons

| This project | ASP.NET Core / EF Core |
| --- | --- |
| main.py and APIRouter modules | Program.cs and controllers/route groups |
| schemas.py Pydantic DTOs | Request/response DTOs and validation |
| models.py Document and Base.metadata | Entity mapping and EF model |
| SQLAlchemy engine and pool | Shared database infrastructure / connection pool |
| Session | DbContext: identity map, change tracking, unit of work |
| get_session yield dependency | Scoped DbContext with disposal after request |
| get_document_service with Depends(get_session) | Scoped service receiving DbContext |
| session.add + session.commit | Add + SaveChanges (explicit transaction commit here) |
| session.get / select | Find / LINQ query |
| Alembic revision --autogenerate | dotnet ef migrations add |
| Alembic upgrade head | dotnet ef database update |

`DocumentService` takes a Session and directly queries ORM entities. It maps them
explicitly to the unchanged Pydantic responses. The router owns HTTP status codes
and 404 handling. No generic repository is needed.

The engine/session factory live for the process. Each request gets a new Session
and DocumentService. FastAPI reuses dependency results within that request, not
across requests. A Session must not be shared across concurrent requests. Sync
handlers/dependencies use FastAPI's worker threads; there is no async database I/O.

`create()` owns its transaction: commit on success, rollback and re-raise on failure.
Commit flushes the INSERT and populates its UUID. `expire_on_commit=False` permits
response mapping without an extra reload. The yield dependency never commits;
closing the session releases connections and rolls back unfinished transactions,
including read transactions. Successful writes commit before the HTTP response.

PostgreSQL now owns storage: data survives API restart/reload and is shared between
API workers. Database/container storage durability depends on its configured volume;
do not remove the database or its data volume if you want to retain documents.
The existing container's bootstrap credentials are used locally; a dedicated limited
application role is a later deployment concern.

## Migrations and verification

The initial migration creates documents with UUID primary key, VARCHAR(200) title,
and TEXT content, all non-null. API startup does not create tables.
For future entity changes, generate a migration and review it before applying:

```powershell
python -m alembic revision --autogenerate -m "describe change"
python -m alembic upgrade head
python -m alembic check
python -m unittest -v test_documents
```

Tests use configured PostgreSQL at Alembic head, not SQLite. They verify separate
request creation/retrieval/listing, fresh-process persistence, 422 validation, 404,
failed-write rollback and session cleanup. They remove only the rows they create.
Do not point integration tests at a production database.

## PATCH and DELETE

`PATCH /documents/{document_id}` accepts a partial object, for example
`{"title": "Updated policy"}`. Omitted fields remain unchanged. Explicit null,
empty strings, and titles longer than 200 characters return 422. An empty object
is a no-op: 200 with the current document, or 404 if absent. Unknown fields are
ignored, consistent with the existing DTOs; they cannot change the document id.
This is a partial-object endpoint, not an RFC 6902 JSON Patch operation array.

`DocumentPatch` uses optional defaults to allow omission, a field validator to
reject supplied nulls, and `model_dump(exclude_unset=True)` to select only supplied
fields. This is a presence-aware update DTO, unlike a C# nullable property alone
which usually cannot distinguish missing JSON from explicit null. No response
DTO or ORM schema changes are needed.

The service loads a tracked entity, applies selected properties, and commits,
like EF Core property assignment followed by SaveChanges. Failed writes roll back.
An empty patch performs no write. Session cleanup remains in the yield dependency.

`DELETE /documents/{document_id}` commits deletion before returning a bodyless
204, comparable to Remove + SaveChanges + NoContent. Missing documents, including
repeated deletion, return 404 with the existing Document not found detail.
Invalid UUIDs return 422. No migration is required.

Integration tests cover partial and combined updates, omitted versus null fields,
empty patches, unchanged data after validation failure, cross-request persistence,
bodyless deletion, missing documents, and rollback after a failed update.
