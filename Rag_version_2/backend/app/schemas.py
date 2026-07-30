from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    namespace: str
    chunks_indexed: int
    filename: str


class QueryRequest(BaseModel):
    namespace: str = Field(..., description="Namespace returned from a prior /upload call")
    question: str = Field(..., min_length=1, description="The question to ask about the document")


class SourceChunk(BaseModel):
    text: str
    source: str
    page: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class ErrorResponse(BaseModel):
    detail: str
