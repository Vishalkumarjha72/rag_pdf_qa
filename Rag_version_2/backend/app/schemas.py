from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    namespace: str
    chunks_indexed: int
    filename: str


class QueryRequest(BaseModel):
    namespace: str = Field(..., description="Namespace returned from a prior /upload call")
    question: str = Field(..., min_length=1, description="The question to ask about the document")
    session_id: str | None = Field(
        default=None,
        description=(
            "Identifies an ongoing conversation. Omit on the first question — "
            "the backend generates one and returns it. Send it back on every "
            "follow-up question in the same conversation so history carries over."
        ),
    )


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    session_id: str = Field(
        description="Pass this back on the next /query call to continue this conversation."
    )


class ErrorResponse(BaseModel):
    detail: str
