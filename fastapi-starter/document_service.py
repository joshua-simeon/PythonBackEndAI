import time
from typing import Annotated
from uuid import UUID

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import get_session
from models import Document
from schemas import DocumentCreate, DocumentPatch, DocumentResponse
#from helper.memoization import memoize
#from helper.measureit import time_it
from helper.cache_with_ttl import cache_with_ttl
class DocumentService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, request: DocumentCreate) -> DocumentResponse:
        document = Document(title=request.title, content=request.content)
        try:
            self._session.add(document)
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        return self._response(document)

    def list(self) -> list[DocumentResponse]:
        documents = self._session.scalars(select(Document)).all()
        return [self._response(document) for document in documents]
    #@memoize
    #@time_it
    @cache_with_ttl(seconds=30.0)
    def get(self, document_id: UUID) -> DocumentResponse | None:
        document = self._session.get(Document, document_id)
        time.sleep(2)  # Simulate a delay for demonstration purposes
        return self._response(document) if document is not None else None

    def update(self, document_id: UUID, request: DocumentPatch) -> DocumentResponse | None:
        document = self._session.get(Document, document_id)
        if document is None:
            return None
        changes = request.model_dump(exclude_unset=True)
        if changes:
            try:
                for field, value in changes.items():
                    setattr(document, field, value)
                self._session.commit()
            except Exception:
                self._session.rollback()
                raise
        return self._response(document)

    def delete(self, document_id: UUID) -> bool:
        document = self._session.get(Document, document_id)
        if document is None:
            return False
        try:
            self._session.delete(document)
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise
        return True

    @staticmethod
    def _response(document: Document) -> DocumentResponse:
        return DocumentResponse(id=document.id, title=document.title, content=document.content)


def get_document_service(
    session: Annotated[Session, Depends(get_session)],
) -> DocumentService:
    return DocumentService(session)
