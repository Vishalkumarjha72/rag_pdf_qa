"""
Uploads eval/dataset.json to LangSmith as a Dataset — a one-time (or
re-run-when-the-dataset-changes) step. langsmith.evaluate()'s `data`
parameter needs real uploaded Examples, not a raw local list (verified
against the installed langsmith==0.10.13 API before writing this).

Every dataset entry — single_turn OR multi_turn — is uploaded in the SAME
shape: a list of turns. A single_turn entry is just a turns list of length
1. This means the eval harness (run_eval.py) and evaluators never need to
branch on entry type — they always iterate `example.outputs["turns"]` and
`run.outputs["turns"]` in lockstep, regardless of length.

Run from inside backend/: python -m eval.upload_dataset
"""

import json
import pathlib

from dotenv import load_dotenv
load_dotenv()  # standalone script, doesn't go through main.py's load_dotenv() call

from langsmith import Client

DATASET_PATH = pathlib.Path(__file__).parent / "dataset.json"
DATASET_NAME = "rag-pdf-qa-eval-v1"

# Every question in dataset.json targets this document — set to whatever
# namespace you currently have this PDF ingested under.
NAMESPACE = "environmental-pollution-38b604eb"


def normalize_entry(entry: dict) -> tuple[dict, dict]:
    """
    Converts one dataset.json entry (either shape) into the uniform
    {"turns": [...]} shape used for BOTH LangSmith inputs and outputs.
    """
    if entry["type"] == "single_turn":
        turns = [{"question": entry["question"]}]
        reference_turns = [{"reference_answer": entry["reference_answer"]}]
    else:  # multi_turn
        turns = [{"question": t["question"]} for t in entry["turns"]]
        reference_turns = [{"reference_answer": t["reference_answer"]} for t in entry["turns"]]

    inputs = {"namespace": NAMESPACE, "turns": turns}
    outputs = {"turns": reference_turns}
    return inputs, outputs


def main():
    with open(DATASET_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    client = Client()

    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset = existing[0]
        print(f"Dataset '{DATASET_NAME}' already exists (id={dataset.id}) — deleting old examples first.")
        for example in client.list_examples(dataset_id=dataset.id):
            client.delete_example(example.id)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Q&A pairs against Environmental Pollution.pdf, for RAG pipeline evaluation (V4).",
        )
        print(f"Created dataset '{DATASET_NAME}' (id={dataset.id})")

    for entry in entries:
        inputs, outputs = normalize_entry(entry)
        client.create_example(
            dataset_id=dataset.id,
            inputs=inputs,
            outputs=outputs,
            metadata={"id": entry["id"], "topic": entry["topic"], "type": entry["type"]},
        )

    print(f"Uploaded {len(entries)} examples to '{DATASET_NAME}'.")


if __name__ == "__main__":
    main()
