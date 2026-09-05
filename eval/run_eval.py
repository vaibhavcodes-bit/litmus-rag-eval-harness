import argparse
import json
from pathlib import Path

from src.retrieval.retriever import retrieve_documents
from eval.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    context_precision,
    average,
)


DATASET_PATH = Path("eval/golden_dataset.json")
RESULTS_DIR = Path("eval/results")


def load_dataset():
    """Load the golden evaluation dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_question(question_data, k=4):
    """Evaluate retrieval for one golden question."""

    question_id = question_data["id"]
    question = question_data["question"]
    expected_document = question_data["reference_document"]

    documents = retrieve_documents(
        question=question,
        k=k,
    )

    retrieved_documents = []

    for document in documents:
        retrieved_documents.append(
            {
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "content": document.page_content,
            }
        )

    expected_sources = []

    if expected_document:
        expected_sources.append(expected_document)

    hit_rate = hit_rate_at_k(
        retrieved_documents=retrieved_documents,
        expected_sources=expected_sources,
        k=k,
    )

    mrr = mean_reciprocal_rank(
        retrieved_documents=retrieved_documents,
        expected_sources=expected_sources,
    )

    precision = context_precision(
        retrieved_documents=retrieved_documents,
        expected_sources=expected_sources,
    )

    return {
        "id": question_id,
        "question": question,
        "category": question_data.get("category"),
        "type": question_data.get("type"),
        "difficulty": question_data.get("difficulty"),
        "expected_document": expected_document,
        "retrieved_documents": retrieved_documents,
        "hit_rate_at_k": hit_rate,
        "mrr": mrr,
        "context_precision": precision,
    }


def run_evaluation(version="v1", k=4):
    """Run retrieval evaluation across the golden dataset."""

    dataset = load_dataset()

    questions = dataset["questions"]

    results = []

    print("=" * 60)
    print(f"Litmus Evaluation - {version}")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print(f"Retrieval K: {k}")
    print()

    for index, question_data in enumerate(
        questions,
        start=1,
    ):
        question_id = question_data["id"]

        print(
            f"[{index}/{len(questions)}] "
            f"Evaluating {question_id}..."
        )

        result = evaluate_question(
            question_data=question_data,
            k=k,
        )

        results.append(result)

    hit_rates = [
        result["hit_rate_at_k"]
        for result in results
    ]

    mrr_scores = [
        result["mrr"]
        for result in results
    ]

    precision_scores = [
        result["context_precision"]
        for result in results
    ]

    summary = {
        "total_questions": len(results),
        "hit_rate_at_k": average(hit_rates),
        "mrr": average(mrr_scores),
        "context_precision": average(precision_scores),
    }

    output = {
        "version": version,
        "retrieval_k": k,
        "summary": summary,
        "questions": results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RESULTS_DIR / f"{version}_results.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("Evaluation complete")
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print()
    print(
        f"Hit Rate@{k}: "
        f"{summary['hit_rate_at_k']:.4f}"
    )
    print(
        f"MRR: "
        f"{summary['mrr']:.4f}"
    )
    print(
        f"Context Precision: "
        f"{summary['context_precision']:.4f}"
    )

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run Litmus RAG evaluation."
    )

    parser.add_argument(
        "--version",
        default="v1",
        help="Evaluation version, for example v1.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of documents to retrieve.",
    )

    args = parser.parse_args()

    run_evaluation(
        version=args.version,
        k=args.k,
    )


if __name__ == "__main__":
    main()