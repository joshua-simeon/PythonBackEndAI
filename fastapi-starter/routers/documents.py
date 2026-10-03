from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from document_service import DocumentService, get_document_service
from schemas import DocumentCreate, DocumentResponse

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse,
             status_code=status.HTTP_201_CREATED)
def create_document(
    request: DocumentCreate,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    return service.create(request)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> list[DocumentResponse]:
    return service.list()


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    document = service.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document
