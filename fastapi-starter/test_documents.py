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
            self.assertIn(document, response.json())

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
            self.assertIn(expected, client.get("/documents").json())
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
            self.assertNotIn(document, client.get("/documents").json())
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


if __name__ == "__main__":
    unittest.main()
