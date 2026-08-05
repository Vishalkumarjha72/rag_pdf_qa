"""
V6 Step 5: retrieval-focused evaluators, separate from eval/evaluators.py's
end-to-end (answer-level) correctness/groundedness/citation checks. These
score the RETRIEVED CHUNKS themselves, before generation ever happens —
the point is to isolate whether V6's hybrid retrieval + reranking changes
actually improved what gets found, independent of how the LLM phrases the
final answer.

Honest limitation: eval/dataset.json's reference data is {question,
reference_answer} pairs, not hand-labeled "these exact chunk indices are
answer-bearing" ground truth. Building real recall@k / precision@k needs
that labeling, which doesn't exist for this dataset yet. Rather than fake
numbers against labels that aren't there, retrieval_hit_rate_evaluator
below is an LLM-judged PROXY for hit rate: "do the retrieved chunks
contain enough information to support the reference answer?" — judged
against the reference answer, not the model's generated one (that's what
separates this from groundedness_evaluator in evaluators.py, which grades
the PREDICTED answer against its own sources). If you hand-label
answer-bearing chunk indices later, swap this for real recall@k/precision@k
using retrieved_chunk_indices below as the raw material.
"""

import logging

from pydantic import BaseModel, Field

from app.retrieval import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class _RetrievalHitGrade(BaseModel):
    contains_answer: bool = Field(
        description=(
            "Whether the retrieved chunks, taken together, contain enough information "
            "to fully support the reference answer — regardless of what the model actually "
            "generated."
        )
    )


_hit_rate_llm = None


def _get_hit_rate_llm():
    global _hit_rate_llm
    if _hit_rate_llm is None:
        _hit_rate_llm = get_llm().with_structured_output(_RetrievalHitGrade)
    return _hit_rate_llm


def _grade_retrieval_hit(question: str, reference_answer: str, sources: list[dict]) -> bool:
    if not sources:
        return False

    context = "\n\n".join(chunk["text"] for chunk in sources)
    llm = _get_hit_rate_llm()
    result = llm.invoke([
        SystemMessage(content=(
            "You are grading a document retrieval system, NOT a generated answer. Given a "
            "question, a known-correct reference answer, and the chunks that were retrieved "
            "for the question, judge only whether the retrieved chunks contain enough "
            "information to construct the reference answer. Ignore phrasing quality entirely."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Reference answer: {reference_answer}\n\n"
            f"Retrieved chunks:\n{context}"
        )),
    ])
    return result.contains_answer


def retrieval_hit_rate_evaluator(run, example) -> dict:
    """
    Fraction of turns where the retrieved chunks (run.outputs) contained
    enough information to support the REFERENCE answer (example.outputs),
    independent of what the model actually generated. This is what V6's
    hybrid retrieval + reranking changes are meant to move — a low answer
    correctness score with a HIGH hit rate here would point at a
    generation-prompt problem, not a retrieval problem; a low hit rate
    points squarely at retrieval.
    """
    predicted_turns = run.outputs["turns"]
    reference_turns = example.outputs["turns"]
    questions = example.inputs["turns"]

    scores = [
        _grade_retrieval_hit(q["question"], ref["reference_answer"], pred["sources"])
        for q, pred, ref in zip(questions, predicted_turns, reference_turns)
    ]
    return {"key": "retrieval_hit_rate", "score": sum(scores) / len(scores)}


def avg_chunks_retrieved_evaluator(run, example) -> dict:
    """
    Sanity-check metric, not a quality score: average number of chunks
    actually returned per turn. Retrieval bugs (e.g. an empty BM25 corpus,
    a broken fusion step) tend to show up here as a sudden drop toward 0
    before they show up in the harder-to-diagnose correctness/groundedness
    numbers.
    """
    predicted_turns = run.outputs["turns"]
    counts = [len(pred["sources"]) for pred in predicted_turns]
    return {"key": "avg_chunks_retrieved", "score": sum(counts) / len(counts)}
