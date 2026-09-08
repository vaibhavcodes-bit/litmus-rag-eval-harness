import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


# Add the project root to Python's import path.
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from dotenv import load_dotenv

from src.pipeline import answer_question

load_dotenv()


ROOT_DIR = Path(__file__).resolve().parents[1]

CACHE_PATH = ROOT_DIR / "eval" / "results" / "v1_generation_cache.json"
RESULTS_DIR = ROOT_DIR / "eval" / "results"
V6_RESULTS_PATH = RESULTS_DIR / "v6_results.json"
V6_REPORT_PATH = RESULTS_DIR / "v6_report.md"


def load_questions() -> List[Dict[str, Any]]:
    """Load the 50-question golden dataset from the existing V1 cache."""

    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Golden question cache not found: {CACHE_PATH}"
        )

    with CACHE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    questions = data.get("questions")

    if not isinstance(questions, list):
        raise ValueError(
            "Expected 'questions' to be a list in v1_generation_cache.json"
        )

    return questions


def get_reference_source(question: Dict[str, Any]) -> str:
    """Return the expected reference document for a question."""

    reference_document = question.get("reference_document")

    if not reference_document:
        return ""

    return str(reference_document).strip()


def source_matches_reference(
    documents: List[Dict[str, Any]],
    reference_document: str,
) -> bool:
    """Check whether the expected reference document was retrieved."""

    if not reference_document:
        return False

    expected = Path(reference_document).name.lower()

    for document in documents:
        source = str(document.get("source", "")).strip()

        if Path(source).name.lower() == expected:
            return True

    return False


def reciprocal_rank(
    documents: List[Dict[str, Any]],
    reference_document: str,
) -> float:
    """
    Calculate reciprocal rank of the expected reference document.

    Returns:
        1/rank if found, otherwise 0.
    """

    if not reference_document:
        return 0.0

    expected = Path(reference_document).name.lower()

    for rank, document in enumerate(documents, start=1):
        source = str(document.get("source", "")).strip()

        if Path(source).name.lower() == expected:
            return 1.0 / rank

    return 0.0


def calculate_context_precision(
    documents: List[Dict[str, Any]],
    reference_document: str,
) -> float:
    """
    Calculate a simple source-level context precision.

    A retrieved document is considered relevant when its source matches
    the expected reference document.
    """

    if not documents:
        return 0.0

    if not reference_document:
        return 0.0

    expected = Path(reference_document).name.lower()

    relevant_count = 0

    for document in documents:
        source = str(document.get("source", "")).strip()

        if Path(source).name.lower() == expected:
            relevant_count += 1

    return relevant_count / len(documents)


def evaluate_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """Run one golden question through V6."""

    question_id = question.get("id", "")
    question_text = question.get("question", "")
    ground_truth = question.get("ground_truth", "")
    reference_document = get_reference_source(question)

    if not question_text:
        raise ValueError(f"{question_id}: question is empty")

    print(f"\nEvaluating {question_id}: {question_text}")

    try:
        result = answer_question(
            question_text,
            mode="v6",
        )

        sources = result.get("sources", [])

        hit = source_matches_reference(
            documents=sources,
            reference_document=reference_document,
        )

        rr = reciprocal_rank(
            documents=sources,
            reference_document=reference_document,
        )

        context_precision = calculate_context_precision(
            documents=sources,
            reference_document=reference_document,
        )

        evaluation_result = {
            "id": question_id,
            "question": question_text,
            "category": question.get("category"),
            "difficulty": question.get("difficulty"),
            "ground_truth": ground_truth,
            "reference_document": reference_document,
            "answer": result.get("answer", ""),
            "sources": sources,
            "route": result.get("route"),
            "retrieval_attempts": result.get("retrieval_attempts", 0),
            "rewritten_queries": result.get("rewritten_queries", []),
            "relevant": result.get("relevant", False),
            "hit_rate_at_4": 1.0 if hit else 0.0,
            "mrr": rr,
            "context_precision": context_precision,
            "error": None,
        }

        print(
            f"  Hit@4={evaluation_result['hit_rate_at_4']:.2f} "
            f"MRR={evaluation_result['mrr']:.2f} "
            f"Precision={evaluation_result['context_precision']:.2f} "
            f"Attempts={evaluation_result['retrieval_attempts']} "
            f"Relevant={evaluation_result['relevant']}"
        )

        return evaluation_result

    except Exception as exc:
        print(f"  ERROR: {exc}")

        return {
            "id": question_id,
            "question": question_text,
            "category": question.get("category"),
            "difficulty": question.get("difficulty"),
            "ground_truth": ground_truth,
            "reference_document": reference_document,
            "answer": "",
            "sources": [],
            "route": None,
            "retrieval_attempts": 0,
            "rewritten_queries": [],
            "relevant": False,
            "hit_rate_at_4": 0.0,
            "mrr": 0.0,
            "context_precision": 0.0,
            "error": str(exc),
        }


def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate V6 retrieval statistics."""

    if not results:
        return {
            "question_count": 0,
            "hit_rate_at_4": 0.0,
            "mrr": 0.0,
            "context_precision": 0.0,
            "rewrite_count": 0,
            "rewrite_rate": 0.0,
            "average_retrieval_attempts": 0.0,
            "corrective_successes": 0,
            "corrective_success_rate": 0.0,
        }

    count = len(results)

    hit_rate = sum(
        result["hit_rate_at_4"]
        for result in results
    ) / count

    mrr = sum(
        result["mrr"]
        for result in results
    ) / count

    context_precision = sum(
        result["context_precision"]
        for result in results
    ) / count

    rewrite_count = sum(
        1
        for result in results
        if result.get("rewritten_queries")
    )

    average_attempts = sum(
        result.get("retrieval_attempts", 0)
        for result in results
    ) / count

    corrective_successes = sum(
        1
        for result in results
        if result.get("rewritten_queries")
        and result.get("relevant")
    )

    rewrite_rate = rewrite_count / count

    corrective_success_rate = (
        corrective_successes / rewrite_count
        if rewrite_count
        else 0.0
    )

    return {
        "question_count": count,
        "hit_rate_at_4": hit_rate,
        "mrr": mrr,
        "context_precision": context_precision,
        "rewrite_count": rewrite_count,
        "rewrite_rate": rewrite_rate,
        "average_retrieval_attempts": average_attempts,
        "corrective_successes": corrective_successes,
        "corrective_success_rate": corrective_success_rate,
    }


def write_report(
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    """Write a human-readable V6 evaluation report."""

    lines = [
        "# V6 Corrective RAG Evaluation",
        "",
        "## Summary",
        "",
        f"- Questions: {summary['question_count']}",
        f"- Hit Rate@4: {summary['hit_rate_at_4']:.4f}",
        f"- MRR: {summary['mrr']:.4f}",
        f"- Context Precision: {summary['context_precision']:.4f}",
        f"- Rewrites: {summary['rewrite_count']}",
        f"- Rewrite Rate: {summary['rewrite_rate']:.4f}",
        (
            "- Average Retrieval Attempts: "
            f"{summary['average_retrieval_attempts']:.4f}"
        ),
        f"- Corrective Successes: {summary['corrective_successes']}",
        (
            "- Corrective Success Rate: "
            f"{summary['corrective_success_rate']:.4f}"
        ),
        "",
        "## Questions Requiring Corrective Retrieval",
        "",
    ]

    corrective_results = [
        result
        for result in results
        if result.get("rewritten_queries")
    ]

    if not corrective_results:
        lines.append("No questions required query rewriting.")
    else:
        for result in corrective_results:
            lines.append(
                f"### {result['id']} — {result['question']}"
            )
            lines.append("")
            lines.append(
                f"- Attempts: {result['retrieval_attempts']}"
            )
            lines.append(
                f"- Relevant: {result['relevant']}"
            )
            lines.append(
                f"- Hit@4: {result['hit_rate_at_4']:.2f}"
            )
            lines.append(
                f"- MRR: {result['mrr']:.2f}"
            )
            lines.append(
                f"- Rewritten queries: "
                f"{result['rewritten_queries']}"
            )
            lines.append("")

    lines.extend(
        [
            "## Failed Questions",
            "",
        ]
    )

    failed_results = [
        result
        for result in results
        if result["hit_rate_at_4"] == 0.0
    ]

    if not failed_results:
        lines.append("No retrieval failures.")
    else:
        for result in failed_results:
            lines.append(
                f"- {result['id']}: {result['question']}"
            )

    V6_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    V6_REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V6 Corrective RAG evaluation."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions.",
    )

    parser.add_argument(
        "--id",
        dest="question_id",
        default=None,
        help="Evaluate one specific question by ID, for example Q037.",
    )

    args = parser.parse_args()

    # Load all golden questions first.
    questions = load_questions()

    # Option 1: evaluate one specific question.
    if args.question_id:
        questions = [
            question
            for question in questions
            if question.get("id") == args.question_id
        ]

        if not questions:
            raise ValueError(
                f"Question ID not found: {args.question_id}"
            )

    # Option 2: evaluate only the first N questions.
    elif args.limit is not None:
        if args.limit <= 0:
            raise ValueError(
                "--limit must be greater than zero"
            )

        questions = questions[:args.limit]

    # If neither --id nor --limit is provided,
    # all questions are evaluated.

    print("=" * 70)
    print("V6 CORRECTIVE RAG EVALUATION")
    print("=" * 70)
    print(f"Questions: {len(questions)}")

    results = []

    for question in questions:
        result = evaluate_question(question)
        results.append(result)

    summary = calculate_summary(results)

    output = {
        "version": "v6",
        "summary": summary,
        "questions": results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with V6_RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(
        summary=summary,
        results=results,
    )

    print("\n" + "=" * 70)
    print("V6 EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Hit Rate@4:       {summary['hit_rate_at_4']:.4f}"
    )

    print(
        f"MRR:               {summary['mrr']:.4f}"
    )

    print(
        f"Context Precision: {summary['context_precision']:.4f}"
    )

    print(
        f"Rewrites:          {summary['rewrite_count']}"
    )

    print(
        f"Rewrite Rate:      {summary['rewrite_rate']:.4f}"
    )

    print(
        "Average Attempts:  "
        f"{summary['average_retrieval_attempts']:.4f}"
    )

    print(
        f"Corrective Success: {summary['corrective_successes']}"
    )

    print(
        "Corrective Success Rate: "
        f"{summary['corrective_success_rate']:.4f}"
    )

    print(f"\nResults: {V6_RESULTS_PATH}")
    print(f"Report:  {V6_REPORT_PATH}")


if __name__ == "__main__":
    main()