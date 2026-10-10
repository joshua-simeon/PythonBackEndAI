"""Integration tests: require the configured PostgreSQL database at Alembic head."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import DataError

from database import SessionLocal, get_session
from document_service import DocumentService
from main import app
from models import Document
from schemas import DocumentCreate, DocumentPatch


class DocumentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = []
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.addCleanup(self.cleanup_rows)

    def cleanup_rows(self) -> None:
        with SessionLocal.begin() as session:
            for document_id in self.ids:
                document = session.get(Document, document_id)
                if document is not None:
                    session.delete(document)

    def create_document(self) -> dict:
        response = self.client.post("/documents", json={"title": "Policy", "content": "Training policy"})
        self.assertEqual(response.status_code, 201)
        document = response.json()
        self.ids.append(UUID(document["id"]))
        self.assertEqual(set(document), {"id", "title", "content"})
        return document

    def test_create_list_and_retrieve_across_requests(self) -> None:
        document = self.create_document()
        with TestClient(app) as client:
            response = client.get(f"/documents/{document['id']}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), document)
            response = client.get("/documents")
            self.assertEqual(response.status_code, 200)
            self.assertIn(document, response.json()["items"])

    def test_persistence_in_fresh_api_process(self) -> None:
        document = self.create_document()
        code = (
            "import sys; from fastapi.testclient import TestClient; from main import app; "
            "client = TestClient(app); response = client.get('/documents/' + sys.argv[1]); "
            "assert response.status_code == 200; print(response.text); client.close()"
        )
        # No inherited engine, session, service, or Python memory in this process.
        result = subprocess.run([sys.executable, "-c", code, document["id"]],
                                cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), document)

    def test_validation(self) -> None:
        before = self.client.get("/documents").json()
        for body in ({"title": "", "content": "text"},
                     {"title": "x" * 201, "content": "text"},
                     {"title": "Title", "content": ""}, {"title": "Title"}):
            with self.subTest(body=body):
                response = self.client.post("/documents", json=body)
                self.assertEqual(response.status_code, 422)
                self.assertIn("detail", response.json())
        self.assertEqual(self.client.get("/documents/not-a-uuid").status_code, 422)
        self.assertEqual(self.client.get("/documents").json(), before)

    def test_missing_document(self) -> None:
        response = self.client.get(f"/documents/{uuid4()}")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Document not found"})

    def test_patch_preserves_omitted_fields_and_persists(self) -> None:
        document = self.create_document()
        path = f"/documents/{document['id']}"
        response = self.client.patch(path, json={"title": "Updated policy"})
        self.assertEqual(response.status_code, 200)
        expected = {**document, "title": "Updated policy"}
        self.assertEqual(response.json(), expected)
        with TestClient(app) as client:
            self.assertEqual(client.get(path).json(), expected)
            self.assertIn(expected, client.get("/documents", params={"title": expected["title"]}).json()["items"])
            response = client.patch(path, json={"content": "New content"})
            self.assertEqual(response.status_code, 200)
            expected["content"] = "New content"
            self.assertEqual(response.json(), expected)
        self.assertEqual(self.client.get(path).json(), expected)
        response = self.client.patch(path, json={"title": "x" * 200, "content": "Both updated"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get(path).json(), response.json())

    def test_empty_patch_is_no_op(self) -> None:
        document = self.create_document()
        path = f"/documents/{document['id']}"
        response = self.client.patch(path, json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), document)
        self.assertEqual(self.client.get(path).json(), document)

    def test_patch_validation_does_not_modify_document(self) -> None:
        document = self.create_document()
        path = f"/documents/{document['id']}"
        for body in ({"title": None}, {"content": None}, {"title": ""},
                     {"title": "x" * 201}, {"content": ""},
                     {"title": "Valid", "content": None}, {"title": 123}):
            with self.subTest(body=body):
                self.assertEqual(self.client.patch(path, json=body).status_code, 422)
                self.assertEqual(self.client.get(path).json(), document)
        self.assertEqual(self.client.patch(path).status_code, 422)

    def test_patch_and_delete_missing_or_invalid_id(self) -> None:
        path = f"/documents/{uuid4()}"
        for body in ({"title": "Updated"}, {}):
            response = self.client.patch(path, json=body)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), {"detail": "Document not found"})
        response = self.client.delete(path)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Document not found"})
        self.assertEqual(self.client.patch("/documents/not-a-uuid", json={}).status_code, 422)
        self.assertEqual(self.client.delete("/documents/not-a-uuid").status_code, 422)

    def test_delete_is_bodyless_and_persists(self) -> None:
        document = self.create_document()
        path = f"/documents/{document['id']}"
        response = self.client.delete(path)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(response.content, b"")
        with TestClient(app) as client:
            self.assertEqual(client.get(path).status_code, 404)
            self.assertNotIn(document, client.get("/documents").json()["items"])
            self.assertEqual(client.delete(path).status_code, 404)
        with SessionLocal() as session:
            self.assertIsNone(session.get(Document, UUID(document["id"])))

    def test_failed_patch_rolls_back(self) -> None:
        document = self.create_document()
        with SessionLocal() as session:
            service = DocumentService(session)
            invalid = DocumentPatch.model_construct(title="x" * 201)
            with self.assertRaises(DataError):
                service.update(UUID(document["id"]), invalid)
            self.assertFalse(session.in_transaction())
            self.assertEqual(service.get(UUID(document["id"])).title, document["title"])
        self.assertEqual(self.client.get(f"/documents/{document['id']}").json(), document)

    def test_failed_write_rolls_back(self) -> None:
        with SessionLocal() as session:
            service = DocumentService(session)
            # Bypass DTO validation to exercise a real PostgreSQL write failure.
            invalid = DocumentCreate.model_construct(title="x" * 201, content="text")
            with self.assertRaises(DataError):
                service.create(invalid)
            self.assertFalse(session.in_transaction())
            document = service.create(DocumentCreate(title="After rollback", content="ok"))
            self.ids.append(document.id)
            self.assertIsNotNone(service.get(document.id))

    def test_session_dependency_closes_and_health(self) -> None:
        dependency = get_session()
        session = next(dependency)
        session.connection()
        dependency.close()
        self.assertFalse(session.in_transaction())
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})


class DocumentListTests(DocumentApiTests):
    # Reuse lifecycle helpers without rerunning the CRUD tests.
    def seed(self, suffixes: list[str]) -> tuple[str, list[dict]]:
        marker = str(uuid4())
        documents = []
        for suffix in suffixes:
            response = self.client.post(
                "/documents", json={"title": f"{marker} {suffix}", "content": "list test"},
            )
            self.assertEqual(response.status_code, 201)
            document = response.json()
            self.ids.append(UUID(document["id"]))
            documents.append(document)
        return marker, documents

    def page(self, **params) -> dict:
        response = self.client.get("/documents", params=params)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"items", "total"})
        return response.json()

    def test_page_boundaries_and_total(self) -> None:
        marker, documents = self.seed(["same"] * 5)
        expected = sorted(documents, key=lambda document: document["id"])
        for offset in (0, 2, 4, 5, 10):
            with self.subTest(offset=offset):
                page = self.page(title=marker, limit=2, offset=offset)
                self.assertEqual(page["items"], expected[offset:offset + 2])
                self.assertEqual(page["total"], 5)

    def test_defaults_and_maximum_limit(self) -> None:
        marker, documents = self.seed(["same"] * 21)
        expected = sorted(documents, key=lambda document: document["id"])
        self.assertEqual(self.page(title=marker), {"items": expected[:20], "total": 21})
        self.assertEqual(self.page(title=marker, limit=100), {"items": expected, "total": 21})

    def test_case_insensitive_substring_and_literal_wildcards(self) -> None:
        marker, documents = self.seed(["Alpha", "aLPHa", "Beta", "100%_done", "100XYdone"])
        page = self.page(title=f"{marker} ALPH", limit=1)
        self.assertEqual(page["total"], 2)
        self.assertIn(page["items"][0], documents[:2])
        self.assertEqual(self.page(title=f"{marker} 100%_"), {"items": [documents[3]], "total": 1})
        self.assertEqual(self.page(title=f"{marker} absent"), {"items": [], "total": 0})

    def test_sorting_and_unique_tie_breaker(self) -> None:
        marker, documents = self.seed(["Beta", "Alpha", "Alpha", "Gamma"])
        for field in ("id", "title"):
            for order in ("asc", "desc"):
                with self.subTest(field=field, order=order):
                    expected = sorted(documents, key=lambda document: document["id"])
                    expected = sorted(expected, key=lambda document: document[field], reverse=order == "desc")
                    pages = [self.page(title=marker, sort_by=field, sort_order=order,
                                       limit=1, offset=offset) for offset in range(4)]
                    self.assertEqual([page["items"][0] for page in pages], expected)
                    self.assertTrue(all(page["total"] == 4 for page in pages))

    def test_invalid_list_parameters(self) -> None:
        for params in ({"limit": 0}, {"limit": -1}, {"limit": 101},
                       {"limit": "abc"}, {"limit": "1.5"}, {"offset": -1},
                       {"offset": "abc"}, {"offset": "1.5"},
                       {"sort_by": "content"}, {"sort_by": "title; DROP TABLE documents"},
                       {"sort_order": "invalid"}):
            with self.subTest(params=params):
                self.assertEqual(self.client.get("/documents", params=params).status_code, 422)


if __name__ == "__main__":
    unittest.main()
