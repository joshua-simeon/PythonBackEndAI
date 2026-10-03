from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status

from document_service import DocumentService, get_document_service
from schemas import DocumentCreate, DocumentPatch, DocumentResponse

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


@router.patch("/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: UUID,
    request: DocumentPatch,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    document = service.update(document_id, request)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_class=Response)
def delete_document(
    document_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> Response:
    if not service.delete(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
