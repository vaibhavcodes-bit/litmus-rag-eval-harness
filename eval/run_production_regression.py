import json
import time
from pathlib import Path

import requests


API_URL = "https://litmus-rag-api-516781219845.asia-south1.run.app/ask"
DATASET_PATH = Path("eval/golden_dataset.json")
RESULT_PATH = Path("test-results/t09-production-regression.json")


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    questions = dataset["questions"]
    results = []

    passed = 0
    failed = 0

    print("=" * 60)
    print("T09 - Production Golden Regression")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print(f"API: {API_URL}")
    print("Mode: v1")
    print("K: 4")
    print()

    for index, item in enumerate(questions, start=1):
        question_id = item["id"]
        question = item["question"]
        expected_source = item["reference_document"]

        started = time.perf_counter()

        try:
            response = requests.post(
                API_URL,
                json={
                    "question": question,
                    "k": 4,
                    "mode": "v1",
                },
                timeout=120,
            )

            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            response.raise_for_status()
            payload = response.json()

            sources = [
                source.get("source")
                for source in payload.get("sources", [])
            ]

            if expected_source is None:
                test_passed = True
            else:
                test_passed = expected_source in sources[:4]

            if test_passed:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            result = {
                "id": question_id,
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": sources[:4],
                "status": status,
                "latency_ms": elapsed_ms,
            }

        except Exception as exc:
            failed += 1

            result = {
                "id": question_id,
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": [],
                "status": "ERROR",
                "error": str(exc),
            }

        results.append(result)

        print(
            f"{question_id} | {status} | "
            f"expected={expected_source} | "
            f"sources={result.get('retrieved_sources', [])}"
        )

    total = len(questions)

    summary = {
        "test": "T09",
        "name": "Production Golden Regression",
        "api_url": API_URL,
        "mode": "v1",
        "k": 4,
        "total_questions": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "results": results,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("T09 COMPLETE")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(
        f"Pass Rate: "
        f"{(passed / total * 100):.2f}%"
        if total
        else "Pass Rate: 0%"
    )
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
