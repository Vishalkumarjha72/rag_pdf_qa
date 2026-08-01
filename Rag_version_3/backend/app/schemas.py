from typing import Literal

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


class AnswerMetadata(BaseModel):
    """
    V3: structured, Pydantic-validated self-assessment the model produces
    about its OWN answer, via a second llm.with_structured_output() call
    made after the answer text itself is generated (see
    generate_answer_metadata() in retrieval.py). Unlike the free-form
    answer, this is guaranteed to always match this exact shape —
    confidence can only ever be one of the three literal values, and
    cited_chunk_indices is always a real list of ints, never missing or
    malformed — that's the whole point of structured output over just
    asking the model to "mention which sources it used" in prose and
    hoping to parse it back out reliably.
    """

    confidence: Literal["high", "medium", "low"] = Field(
        description="The model's self-assessed confidence that the answer is well-supported by the retrieved context."
    )
    cited_chunk_indices: list[int] = Field(
        description="Indices (0-based, into the sources list) of the chunks the model actually drew on to answer — a subset of all retrieved chunks, not necessarily all of them."
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    metadata: AnswerMetadata
    session_id: str = Field(
        description="Pass this back on the next /query call to continue this conversation."
    )


class ErrorResponse(BaseModel):
    detail: str
