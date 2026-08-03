"""
Row-level evaluators for langsmith.evaluate(). Verified signature against
the installed langsmith==0.10.13: an evaluator is a plain function
`(run: Run, example: Example) -> dict`, returning {"key": ..., "score": ...}.

`run.outputs` is whatever eval/run_eval.py's predict() function returned;
`example.outputs` is what upload_dataset.py stored as the reference. Both
always have the shape {"turns": [...]}, regardless of whether the original
dataset entry was single_turn or multi_turn — see upload_dataset.py's
normalize_entry() for why.
"""

import re

from pydantic import BaseModel, Field

from app.retrieval import get_llm
from langchain_core.messages import SystemMessage, HumanMessage


class _CorrectnessGrade(BaseModel):
    is_correct: bool = Field(
        description="Whether the predicted answer captures the same meaning as the reference answer, even if worded differently."
    )


class _GroundednessGrade(BaseModel):
    is_grounded: bool = Field(
        description="Whether the predicted answer is actually supported by the given source chunks, without inventing information not present in them."
    )


_correctness_llm = None
_groundedness_llm = None


def _get_correctness_llm():
    global _correctness_llm
    if _correctness_llm is None:
        _correctness_llm = get_llm().with_structured_output(_CorrectnessGrade)
    return _correctness_llm


def _get_groundedness_llm():
    global _groundedness_llm
    if _groundedness_llm is None:
        _groundedness_llm = get_llm().with_structured_output(_GroundednessGrade)
    return _groundedness_llm


def _grade_correctness(question: str, predicted_answer: str, reference_answer: str) -> bool:
    llm = _get_correctness_llm()
    result = llm.invoke([
        SystemMessage(content=(
            "You are grading whether a predicted answer is correct compared to a reference "
            "answer, for the given question. Focus on factual meaning, not exact wording."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Reference answer: {reference_answer}\n\n"
            f"Predicted answer: {predicted_answer}"
        )),
    ])
    return result.is_correct


def _grade_groundedness(question: str, predicted_answer: str, sources: list[dict]) -> bool:
    """
    Deliberately does NOT use the reference answer — this checks whether
    the PREDICTED answer is supported by the chunks that were ACTUALLY
    retrieved for this run, independent of whether it also happens to
    match the reference. An answer can be correct AND ungrounded (lucky
    guess / prior knowledge) or grounded AND incorrect (context genuinely
    didn't contain the answer) — that distinction is the whole point of
    grading these separately.
    """
    if not sources:
        # No sources retrieved — the only grounded answer is one that
        # admits it doesn't know. Anything else is definitionally ungrounded.
        return "don't know" in predicted_answer.lower() or "couldn't find" in predicted_answer.lower()

    context = "\n\n".join(chunk["text"] for chunk in sources)
    llm = _get_groundedness_llm()
    result = llm.invoke([
        SystemMessage(content=(
            "You are grading whether a predicted answer is actually supported by the given "
            "source chunks, for the given question. The answer should not include claims that "
            "aren't backed by the sources, even if those claims happen to be true in general."
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Source chunks:\n{context}\n\n"
            f"Predicted answer: {predicted_answer}"
        )),
    ])
    return result.is_grounded


def correctness_evaluator(run, example) -> dict:
    """Average correctness across every turn in this example (1 for single_turn)."""
    predicted_turns = run.outputs["turns"]
    reference_turns = example.outputs["turns"]
    questions = example.inputs["turns"]

    scores = [
        _grade_correctness(q["question"], pred["answer"], ref["reference_answer"])
        for q, pred, ref in zip(questions, predicted_turns, reference_turns)
    ]
    return {"key": "correctness", "score": sum(scores) / len(scores)}


def groundedness_evaluator(run, example) -> dict:
    """Average groundedness across every turn in this example."""
    predicted_turns = run.outputs["turns"]
    questions = example.inputs["turns"]

    scores = [
        _grade_groundedness(q["question"], pred["answer"], pred["sources"])
        for q, pred in zip(questions, predicted_turns)
    ]
    return {"key": "groundedness", "score": sum(scores) / len(scores)}


def citation_presence_evaluator(run, example) -> dict:
    """Scores whether the predicted answer includes explicit source citations."""
    predicted_turns = run.outputs["turns"]

    scores = []
    for pred in predicted_turns:
        answer = pred["answer"]
        has_citation = bool(re.search(r"\[source \d+\]", answer))
        scores.append(1.0 if has_citation else 0.0)

    return {"key": "citation_presence", "score": sum(scores) / len(scores)}


def confidence_correlation_evaluator(run, example) -> dict:
    """
    Checks whether the model's OWN self-reported confidence (V3's
    AnswerMetadata) agrees with an independent groundedness judgment.
    "high"/"medium" confidence is expected to correlate with grounded=True;
    "low" confidence is expected to correlate with grounded=False. Score is
    the fraction of turns where self-reported confidence and independent
    judgment actually agree — a low score here would mean V3's confidence
    field isn't trustworthy, which is worth knowing regardless of how
    correctness/groundedness themselves score.
    """
    predicted_turns = run.outputs["turns"]
    questions = example.inputs["turns"]

    agreements = []
    for q, pred in zip(questions, predicted_turns):
        is_grounded = _grade_groundedness(q["question"], pred["answer"], pred["sources"])
        reported_confidence = pred["metadata"]["confidence"]
        expected_grounded = reported_confidence in ("high", "medium")
        agreements.append(expected_grounded == is_grounded)

    return {"key": "confidence_correlation", "score": sum(agreements) / len(agreements)}
