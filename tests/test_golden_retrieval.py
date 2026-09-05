import json
from pathlib import Path

from src.retrieval.retriever import retrieve_documents


DATASET_PATH = Path("eval/golden_dataset.json")


def load_golden_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def test_golden_dataset_file_exists():
    assert DATASET_PATH.exists()


def test_golden_dataset_contains_50_questions():
    dataset = load_golden_dataset()

    assert dataset["total_questions"] == 50
    assert len(dataset["questions"]) == 50


def test_known_golden_questions_retrieve_expected_documents():
    dataset = load_golden_dataset()

    questions_to_test = [
        "Q001",
        "Q008",
        "Q014",
        "Q025",
        "Q032",
        "Q040",
        "Q049",
    ]

    questions = {
        item["id"]: item
        for item in dataset["questions"]
    }

    for question_id in questions_to_test:
        item = questions[question_id]

        expected_document = item["reference_document"]

        if not expected_document:
            continue

        documents = retrieve_documents(
            question=item["question"],
            k=4,
        )

        assert documents, (
            f"No documents retrieved for {question_id}"
        )

        retrieved_sources = [
            document.metadata.get("source")
            for document in documents
        ]

        assert expected_document in retrieved_sources, (
            f"{question_id} expected "
            f"{expected_document}, but retrieved "
            f"{retrieved_sources}"
        )