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


class EvalSummary(BaseModel):
    question_count: int
    hit_rate_at_4: float
    mrr: float
    context_precision: float
    rewrite_count: int
    rewrite_rate: float
    average_retrieval_attempts: float
    corrective_successes: int
    corrective_success_rate: float


class EvalResponse(BaseModel):
    version: str
    summary: EvalSummary
    questions: list[dict]