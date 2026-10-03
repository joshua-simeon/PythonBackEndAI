from threading import Lock
from uuid import UUID, uuid4

from schemas import DocumentCreate, DocumentResponse


class DocumentService:
    def __init__(self) -> None:
        self._documents: dict[UUID, DocumentResponse] = {}
        self._lock = Lock()

    def create(self, request: DocumentCreate) -> DocumentResponse:
        document = DocumentResponse(id=uuid4(), **request.model_dump())
        with self._lock:
            self._documents[document.id] = document
        return document

    def list(self) -> list[DocumentResponse]:
        with self._lock:
            return list(self._documents.values())

    def get(self, document_id: UUID) -> DocumentResponse | None:
        with self._lock:
            return self._documents.get(document_id)


# One service and store per worker process; both reset on restart/reload.
_document_service = DocumentService()


def get_document_service() -> DocumentService:
    return _document_service
