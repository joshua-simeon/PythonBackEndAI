import unittest
from uuid import uuid4

from fastapi.testclient import TestClient

from document_service import DocumentService, get_document_service
from main import app


class DocumentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = DocumentService()
        app.dependency_overrides[get_document_service] = lambda: self.service
        self.client = TestClient(app)
        self.addCleanup(app.dependency_overrides.pop, get_document_service)
        self.addCleanup(self.client.close)

    def test_create_then_retrieve_in_separate_requests(self) -> None:
        response = self.client.post(
            "/documents", json={"title": "Policy", "content": "Training policy"}
        )
        self.assertEqual(response.status_code, 201)
        document = response.json()
        # A second client also sees the shared service's storage.
        with TestClient(app) as second_client:
            retrieved = second_client.get(f"/documents/{document['id']}")
            self.assertEqual(retrieved.status_code, 200)
            self.assertEqual(retrieved.json(), document)
            self.assertEqual(second_client.get("/documents").json(), [document])

    def test_validation(self) -> None:
        for body in (
            {"title": "", "content": "text"},
            {"title": "x" * 201, "content": "text"},
            {"title": "Title", "content": ""},
            {"title": "Title"},
        ):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/documents", json=body).status_code, 422)
        self.assertEqual(self.client.get("/documents/not-a-uuid").status_code, 422)
        self.assertEqual(self.client.get("/documents").json(), [])

    def test_missing_document(self) -> None:
        response = self.client.get(f"/documents/{uuid4()}")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Document not found"})

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_default_provider_shares_service(self) -> None:
        self.assertIs(get_document_service(), get_document_service())


if __name__ == "__main__":
    unittest.main()
