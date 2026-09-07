import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.router import route_question

DATASET_PATH = (
    PROJECT_ROOT
    / "eval"
    / "v5_golden_dataset.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "eval"
    / "results"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "v5_routing_results.json"
)


def load_dataset():
    """Load the V5 routing golden dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate():
    """Run the V5 routing benchmark."""

    dataset = load_dataset()

    results = []

    correct = 0

    for item in dataset:

        question_id = item["id"]
        question = item["question"]
        expected_route = item["expected_route"]

        print(
            f"Evaluating {question_id}: "
            f"{question}"
        )

        try:

            actual_route = route_question(
                question
            )

            is_correct = (
                actual_route == expected_route
            )

            if is_correct:
                correct += 1

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected_route": expected_route,
                    "actual_route": actual_route,
                    "correct": is_correct,
                    "error": None,
                }
            )

            print(
                f"  Expected: {expected_route}"
            )

            print(
                f"  Actual:   {actual_route}"
            )

            print(
                f"  Correct:  {is_correct}"
            )

        except Exception as error:

            results.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected_route": expected_route,
                    "actual_route": None,
                    "correct": False,
                    "error": str(error),
                }
            )

            print(
                f"  ERROR: {error}"
            )

    total = len(dataset)

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    summary = {
        "total_questions": total,
        "correct_routes": correct,
        "incorrect_routes": total - correct,
        "routing_accuracy": accuracy,
    }

    output = {
        "version": "v5",
        "summary": summary,
        "results": results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
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
    print("V5 ROUTING EVALUATION")
    print("=" * 60)

    print(
        f"Total questions:   {total}"
    )

    print(
        f"Correct routes:    {correct}"
    )

    print(
        f"Incorrect routes:  {total - correct}"
    )

    print(
        f"Routing accuracy:  {accuracy:.4f}"
    )

    print(
        f"Results saved to:  {RESULTS_PATH}"
    )


if __name__ == "__main__":
    evaluate()