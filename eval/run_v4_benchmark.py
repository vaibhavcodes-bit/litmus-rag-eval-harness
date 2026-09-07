import argparse
import json
from pathlib import Path

from eval.v4_benchmark_retriever import (
    build_v4_vector_store,
    retrieve_v4_benchmark,
)

from src.retrieval.decomposition import decompose_question


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "eval"
    / "v4_golden_dataset.json"
)

RESULT_PATH = (
    BASE_DIR
    / "eval"
    / "results"
    / "v4_benchmark_results.json"
)


def load_dataset():
    """Load the dedicated V4 benchmark dataset."""

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_source(document):
    """Return the source filename for a document."""

    return document.metadata.get(
        "source",
        "",
    )


def source_hit(
    documents,
    expected_documents,
):
    """Return 1 if any expected document was retrieved."""

    retrieved_sources = {
        normalize_source(document)
        for document in documents
    }

    return int(
        any(
            expected in retrieved_sources
            for expected in expected_documents
        )
    )


def reciprocal_rank(
    documents,
    expected_documents,
):
    """Calculate reciprocal rank."""

    for rank, document in enumerate(
        documents,
        start=1,
    ):
        source = normalize_source(document)

        if source in expected_documents:
            return 1.0 / rank

    return 0.0


def precision_at_k(
    documents,
    expected_documents,
):
    """Calculate precision among retrieved documents."""

    if not documents:
        return 0.0

    relevant = 0

    for document in documents:
        if (
            normalize_source(document)
            in expected_documents
        ):
            relevant += 1

    return relevant / len(documents)


def evaluate_v3(
    vector_store,
    questions,
    k,
):
    """
    Evaluate the original question directly.

    This represents the V3-style baseline for this
    dedicated benchmark.
    """

    results = []

    for item in questions:

        documents = retrieve_v4_benchmark(
            vector_store,
            item["question"],
            k=k,
        )

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "retrieved_sources": [
                    normalize_source(document)
                    for document in documents
                ],
                "hit_rate_at_k": source_hit(
                    documents,
                    item["expected_documents"],
                ),
                "mrr": reciprocal_rank(
                    documents,
                    item["expected_documents"],
                ),
                "context_precision": precision_at_k(
                    documents,
                    item["expected_documents"],
                ),
            }
        )

    return results


def evaluate_v4(
    vector_store,
    questions,
    k,
):
    """
    Evaluate V4 using independent sub-question retrieval.

    Each sub-question is evaluated separately instead
    of flattening all retrieved documents together.
    """

    results = []

    for item in questions:

        print(
            f"\nProcessing {item['id']}: "
            f"{item['question']}"
        )

        sub_questions = decompose_question(
            item["question"],
            max_sub_questions=4,
        )

        print("  Sub-questions:")

        for sub_question in sub_questions:
            print(
                f"    - {sub_question}"
            )

        sub_question_results = []

        for sub_question in sub_questions:

            documents = retrieve_v4_benchmark(
                vector_store,
                sub_question,
                k=k,
            )

            hit = source_hit(
                documents,
                item["expected_documents"],
            )

            mrr = reciprocal_rank(
                documents,
                item["expected_documents"],
            )

            precision = precision_at_k(
                documents,
                item["expected_documents"],
            )

            sub_question_results.append(
                {
                    "sub_question": sub_question,
                    "retrieved_sources": [
                        normalize_source(document)
                        for document in documents
                    ],
                    "hit_rate_at_k": hit,
                    "mrr": mrr,
                    "context_precision": precision,
                }
            )

        count = len(
            sub_question_results
        )

        if count == 0:
            average_hit = 0.0
            average_mrr = 0.0
            average_precision = 0.0
        else:
            average_hit = (
                sum(
                    x["hit_rate_at_k"]
                    for x in sub_question_results
                )
                / count
            )

            average_mrr = (
                sum(
                    x["mrr"]
                    for x in sub_question_results
                )
                / count
            )

            average_precision = (
                sum(
                    x["context_precision"]
                    for x in sub_question_results
                )
                / count
            )

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "sub_questions": sub_questions,
                "sub_question_results": (
                    sub_question_results
                ),
                "hit_rate_at_k": round(
                    average_hit,
                    4,
                ),
                "mrr": round(
                    average_mrr,
                    4,
                ),
                "context_precision": round(
                    average_precision,
                    4,
                ),
            }
        )

    return results


def summarize(results):
    """Calculate aggregate benchmark metrics."""

    count = len(results)

    if count == 0:
        return {
            "hit_rate_at_k": 0.0,
            "mrr": 0.0,
            "context_precision": 0.0,
        }

    return {
        "hit_rate_at_k": round(
            sum(
                x["hit_rate_at_k"]
                for x in results
            )
            / count,
            4,
        ),
        "mrr": round(
            sum(
                x["mrr"]
                for x in results
            )
            / count,
            4,
        ),
        "context_precision": round(
            sum(
                x["context_precision"]
                for x in results
            )
            / count,
            4,
        ),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LITMUS V4 DEDICATED BENCHMARK")
    print("=" * 70)

    questions = load_dataset()

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"K: {args.k}"
    )

    print(
        "\nBuilding isolated V4 benchmark "
        "vector store..."
    )

    vector_store = (
        build_v4_vector_store()
    )

    print(
        "Vector store ready."
    )

    print(
        "\nRunning V3 baseline..."
    )

    v3_results = evaluate_v3(
        vector_store,
        questions,
        args.k,
    )

    print(
        "V3 complete."
    )

    print(
        "\nRunning V4 decomposition..."
    )

    v4_results = evaluate_v4(
        vector_store,
        questions,
        args.k,
    )

    print(
        "V4 complete."
    )

    output = {
        "benchmark": "v4_dedicated",
        "evaluation_method": (
            "sub_question_aware"
        ),
        "k": args.k,
        "question_count": len(
            questions
        ),
        "v3": {
            "summary": summarize(
                v3_results
            ),
            "questions": v3_results,
        },
        "v4": {
            "summary": summarize(
                v4_results
            ),
            "questions": v4_results,
        },
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "\nV3 BASELINE"
    )

    print(
        json.dumps(
            summarize(
                v3_results
            ),
            indent=2,
        )
    )

    print(
        "\nV4 DECOMPOSITION"
    )

    print(
        json.dumps(
            summarize(
                v4_results
            ),
            indent=2,
        )
    )

    print(
        f"\nSaved to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()