from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system.",
    )
    k: int = Field(
        default=4,
        ge=1,
        description="Number of documents to retrieve.",
    )
    mode: str = Field(
        default="v1",
        description="RAG pipeline mode.",
    )


class Source(BaseModel):
    source: str | None = None
    page: int | None = None
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class EvalResponse(BaseModel):
    version: str
    summary: dict[str, Any]
    questions: list[dict[str, Any]]