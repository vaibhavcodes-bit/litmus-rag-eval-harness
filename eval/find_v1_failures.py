import json
import sys
from pathlib import Path


# Add project root to Python import path.
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.retrieval.retriever import retrieve_documents


CACHE_PATH = (
    ROOT_DIR
    / "eval"
    / "results"
    / "v1_generation_cache.json"
)


def main():
    # Load the existing 50-question dataset/cache.
    with CACHE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data["questions"]

    answerable_questions = [
        question
        for question in questions
        if question.get("reference_document")
    ]

    failures = []

    print("=" * 70)
    print("V1 RETRIEVAL FAILURE DIAGNOSTIC")
    print("=" * 70)
    print(
        f"Answerable questions: {len(answerable_questions)}"
    )

    for question in answerable_questions:
        question_id = question["id"]
        question_text = question["question"]
        reference_document = question["reference_document"]

        print(f"\nChecking {question_id}: {question_text}")

        documents = retrieve_documents(
            question_text,
            k=4,
        )

        retrieved_sources = []

        for document in documents:
            source = document.metadata.get("source", "")
            retrieved_sources.append(str(source))

        expected_source = Path(
            reference_document
        ).name.lower()

        normalized_sources = [
            Path(source).name.lower()
            for source in retrieved_sources
            if source
        ]

        hit = expected_source in normalized_sources

        print(
            f"  Expected:  {reference_document}"
        )
        print(
            f"  Retrieved: {retrieved_sources}"
        )
        print(
            f"  Hit@4:     {'YES' if hit else 'NO'}"
        )

        if not hit:
            failures.append(
                {
                    "id": question_id,
                    "question": question_text,
                    "expected_source": reference_document,
                    "retrieved_sources": retrieved_sources,
                }
            )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Answerable questions checked: "
        f"{len(answerable_questions)}"
    )

    print(
        f"V1 retrieval failures: {len(failures)}"
    )

    if failures:
        print("\nFAILED QUESTIONS:\n")

        for failure in failures:
            print(
                f"{failure['id']} | "
                f"{failure['question']}"
            )

            print(
                f"  Expected:  "
                f"{failure['expected_source']}"
            )

            print(
                f"  Retrieved: "
                f"{failure['retrieved_sources']}"
            )

    else:
        print("\nNo V1 retrieval failures found.")


if __name__ == "__main__":
    main()