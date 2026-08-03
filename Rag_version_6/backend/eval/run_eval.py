"""
Runs the full V4 evaluation: every example in the uploaded LangSmith
dataset gets run through the REAL graph (app.graph.ask_with_memory), then
scored by every evaluator in eval/evaluators.py.

Uses aevaluate() (not evaluate()) because ask_with_memory is async — see
app/graph.py's module docstring for why the whole graph had to become
async in V3 (AsyncRedisSaver).

Run from inside backend/ (after uploading the dataset at least once):
    python -m eval.run_eval
"""

import asyncio
import uuid

from dotenv import load_dotenv
load_dotenv()  # standalone script, doesn't go through main.py's load_dotenv() call

from langsmith.evaluation import aevaluate

from app.graph import ask_with_memory
from eval.evaluators import (
    correctness_evaluator,
    groundedness_evaluator,
    confidence_correlation_evaluator,
    citation_presence_evaluator,
)
from eval.upload_dataset import DATASET_NAME


async def predict(inputs: dict) -> dict:
    """
    The target function aevaluate() runs against every dataset example.

    A fresh session_id per example (NOT per turn) — for a multi_turn
    example, all its turns run as ONE real conversation sharing a
    session_id, exercising the actual condense/retrieve/generate + Redis
    memory path exactly as a real user's multi-turn conversation would,
    rather than faking history in the input.
    """
    session_id = str(uuid.uuid4())
    namespace = inputs["namespace"]

    results = []
    for turn in inputs["turns"]:
        result = await ask_with_memory(turn["question"], namespace, session_id)
        results.append(result)

    return {"turns": results}


async def main():
    results = await aevaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[
            correctness_evaluator,
            groundedness_evaluator,
            confidence_correlation_evaluator,
            citation_presence_evaluator,
        ],
        experiment_prefix="v5-prompting",
        description="V5 prompting evaluation: correctness, groundedness, confidence correlation, and citation presence against Environmental Pollution.pdf.",
        max_concurrency=1,  # sequential — avoids concurrent calls racing the singleton embedding model from multiple threads at once
    )
    print("\nEvaluation complete. View full results in the LangSmith UI.")
    return results


if __name__ == "__main__":
    asyncio.run(main())
