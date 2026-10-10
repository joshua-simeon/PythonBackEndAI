from uuid import UUID
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class DocumentResponse(DocumentCreate):
    id: UUID


DocumentSortField = Literal["id", "title"]
SortOrder = Literal["asc", "desc"]


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class DocumentPatch(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, min_length=1)

    @field_validator("title", "content")
    @classmethod
    def reject_explicit_null(cls, value: str | None) -> str:
        # Defaults for omitted fields are not validated; supplied nulls are.
        if value is None:
            raise ValueError("Document fields cannot be null")
        return value
