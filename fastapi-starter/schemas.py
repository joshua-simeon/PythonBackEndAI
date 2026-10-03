from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)


class DocumentResponse(DocumentCreate):
    id: UUID


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
