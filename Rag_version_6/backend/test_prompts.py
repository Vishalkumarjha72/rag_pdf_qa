"""Prompt validation tests for V5.

Run from inside backend/: python test_prompts.py
"""

from app.graph import _is_bad_answer
from app.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE


def test_system_prompt_contains_guardrails():
    assert "only the provided document excerpts" in SYSTEM_PROMPT
    assert "If the answer is not contained in the provided context" in SYSTEM_PROMPT
    assert "Citations are required" in SYSTEM_PROMPT


def test_answer_prompt_contains_citation_instruction():
    prompt_text = ANSWER_PROMPT_TEMPLATE.format(context="Example context.", question="What is this?")
    assert "Answer the question using only the context above." in prompt_text
    assert "Cite any facts with source references" in prompt_text
    assert "[source 1]" in prompt_text


def test_fallback_detects_bad_answers():
    assert _is_bad_answer("", [{"text": "Sample chunk"}])
    assert _is_bad_answer("too short", [{"text": "Sample chunk"}])
    assert not _is_bad_answer("This answer is clearly supported by the context.", [{"text": "Sample chunk"}])


def main():
    test_system_prompt_contains_guardrails()
    test_answer_prompt_contains_citation_instruction()
    test_fallback_detects_bad_answers()
    print("All prompt tests passed.")


if __name__ == "__main__":
    main()
