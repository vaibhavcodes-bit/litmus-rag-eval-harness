import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
)

from eval.metrics import refusal_accuracy
from src.pipeline import answer_question
from src.retrieval.embedder import get_embeddings


DATASET_PATH = PROJECT_ROOT / "eval" / "golden_dataset.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Cache file used to save generation progress.
# This allows the evaluation to resume after an API error,
# rate limit, or process interruption.
CACHE_PATH = RESULTS_DIR / "v1_generation_cache.json"


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_cache(question_results):
    """
    Save generation progress immediately.

    The cache is written after every question so that
    successful results are not lost if the process stops.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CACHE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "version": "v1",
                "questions": question_results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_cache():
    """
    Load previously saved generation results.

    Returns an empty list when no cache exists.
    """

    if not CACHE_PATH.exists():
        return []

    try:
        with open(
            CACHE_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        questions = data.get(
            "questions",
            [],
        )

        print(
            f"Loaded {len(questions)} "
            f"cached question results."
        )

        return questions

    except Exception as error:
        print(
            f"WARNING: Could not load cache: {error}"
        )

        print(
            "Starting with an empty cache."
        )

        return []


def build_ragas_dataset(question_results):
    rows = []

    for result in question_results:
        rows.append(
            {
                "user_input": result["question"],
                "response": result["answer"],
                "retrieved_contexts": result[
                    "retrieved_contexts"
                ],
                "reference": result["ground_truth"],
            }
        )

    return Dataset.from_list(rows)


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def create_ragas_llm():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    # Groq provides an OpenAI-compatible API.
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # Ragas evaluator LLM
    ragas_llm = llm_factory(
        model="openai/gpt-oss-20b",
        provider="openai",
        client=client,
        max_tokens=2048,
    )

    return ragas_llm


def run_generation_evaluation(
    version="v1",
    k=4,
    limit=None,
):
    dataset = load_dataset()
    questions = dataset["questions"]

    print("=" * 60)
    print(
        f"Litmus Generation Evaluation - {version}"
    )
    print("=" * 60)

    print(
        f"Questions in golden dataset: {len(questions)}"
    )

    print(
        f"Retrieval K: {k}"
    )

    if limit is not None:
        print(
            f"Question limit for this run: {limit}"
        )

    print()

    # ---------------------------------------------------------
    # Load previously saved results
    # ---------------------------------------------------------

    cached_results = load_cache()

    # Store results by question ID.
    # This prevents duplicate generation.
    question_results_by_id = {
        result["id"]: result
        for result in cached_results
    }

    print(
        f"Cached results available: "
        f"{len(question_results_by_id)}"
    )

    print()

    # ---------------------------------------------------------
    # Decide which questions to process
    # ---------------------------------------------------------

    questions_to_run = questions

    if limit is not None:
        questions_to_run = questions[:limit]

    # ---------------------------------------------------------
    # Generate answers
    # ---------------------------------------------------------

    for index, question_data in enumerate(
        questions_to_run,
        start=1,
    ):
        question_id = question_data["id"]
        question = question_data["question"]

        # -----------------------------------------------------
        # Skip successfully completed questions
        # -----------------------------------------------------

        existing_result = question_results_by_id.get(
            question_id
        )

        if (
            existing_result is not None
            and existing_result.get("error") is None
            and existing_result.get("answer")
        ):
            print(
                f"[{index}/{len(questions_to_run)}] "
                f"{question_id} already completed → SKIP"
            )
            continue

        print(
            f"[{index}/{len(questions_to_run)}] "
            f"Generating answer for {question_id}..."
        )

        try:
            result = answer_question(
                question=question,
                k=k,
            )

            answer = result["answer"]

            retrieved_contexts = [
                source["content"]
                for source in result["sources"]
            ]

            should_refuse = (
                question_data.get(
                    "reference_document"
                )
                is None
                or not question_data.get(
                    "reference_context"
                )
            )

            refusal_score = refusal_accuracy(
                answer=answer,
                should_refuse=should_refuse,
            )

            question_result = {
                "id": question_id,
                "question": question,
                "category": question_data.get(
                    "category"
                ),
                "type": question_data.get(
                    "type"
                ),
                "difficulty": question_data.get(
                    "difficulty"
                ),
                "ground_truth": question_data.get(
                    "ground_truth",
                    "",
                ),
                "reference_document": question_data.get(
                    "reference_document"
                ),
                "answer": answer,
                "retrieved_contexts": retrieved_contexts,
                "refusal_accuracy": refusal_score,
                "error": None,
            }

            question_results_by_id[
                question_id
            ] = question_result

            # -------------------------------------------------
            # Save immediately after successful generation
            # -------------------------------------------------

            save_cache(
                list(
                    question_results_by_id.values()
                )
            )

            print(
                f"  Saved {question_id} to cache."
            )

        except Exception as error:
            print(
                f"  ERROR: {error}"
            )

            question_result = {
                "id": question_id,
                "question": question,
                "category": question_data.get(
                    "category"
                ),
                "type": question_data.get(
                    "type"
                ),
                "difficulty": question_data.get(
                    "difficulty"
                ),
                "ground_truth": question_data.get(
                    "ground_truth",
                    "",
                ),
                "reference_document": question_data.get(
                    "reference_document"
                ),
                "answer": "",
                "retrieved_contexts": [],
                "refusal_accuracy": 0.0,
                "error": str(error),
            }

            question_results_by_id[
                question_id
            ] = question_result

            # Save errors too.
            # On the next run, questions with errors
            # will be attempted again.
            save_cache(
                list(
                    question_results_by_id.values()
                )
            )

    # ---------------------------------------------------------
    # Restore original golden dataset order
    # ---------------------------------------------------------

    question_results = []

    for question_data in questions:
        question_id = question_data["id"]

        result = question_results_by_id.get(
            question_id
        )

        if result is not None:
            question_results.append(result)

    # ---------------------------------------------------------
    # Determine successful and failed questions
    # ---------------------------------------------------------

    successful_results = [
        result
        for result in question_results
        if result["error"] is None
        and result["answer"]
    ]

    failed_results = [
        result
        for result in question_results
        if result["error"] is not None
    ]

    print()

    print("=" * 60)
    print("Generation status")
    print("=" * 60)

    print(
        f"Total golden questions: {len(questions)}"
    )

    print(
        f"Successful cached/generated: "
        f"{len(successful_results)}"
    )

    print(
        f"Failed: {len(failed_results)}"
    )

    print()

    # ---------------------------------------------------------
    # Only run Ragas when ALL 50 questions are complete
    # ---------------------------------------------------------

    if len(successful_results) < len(questions):
        print(
            "Generation is not complete yet."
        )

        print(
            "Ragas evaluation will NOT run."
        )

        print(
            f"Progress saved to: {CACHE_PATH}"
        )

        print()

        print(
            "Run the evaluator again after the "
            "API limit/error is resolved."
        )

        return {
            "version": version,
            "status": "generation_incomplete",
            "total_questions": len(questions),
            "successful_questions": len(
                successful_results
            ),
            "failed_questions": len(
                failed_results
            ),
            "cache_path": str(CACHE_PATH),
        }

    # ---------------------------------------------------------
    # Ragas evaluation
    # ---------------------------------------------------------

    print()

    print("=" * 60)
    print("Running Ragas metrics")
    print("=" * 60)

    ragas_dataset = build_ragas_dataset(
        successful_results
    )

    ragas_llm = create_ragas_llm()

    # Required V2 generation metrics
    metrics = [
        Faithfulness(
            llm=ragas_llm
        ),
        AnswerRelevancy(
            llm=ragas_llm
        ),
        AnswerCorrectness(
            llm=ragas_llm
        ),
    ]

    # Use the project's local embedding model.
    embeddings = get_embeddings()

    ragas_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        embeddings=embeddings,
        raise_exceptions=False,
        show_progress=True,
    )

    ragas_scores = (
        ragas_result.to_pandas()
    )

    faithfulness_scores = [
        float(value)
        for value in ragas_scores[
            "faithfulness"
        ]
        if value is not None
    ]

    answer_relevancy_scores = [
        float(value)
        for value in ragas_scores[
            "answer_relevancy"
        ]
        if value is not None
    ]

    answer_correctness_scores = [
        float(value)
        for value in ragas_scores[
            "answer_correctness"
        ]
        if value is not None
    ]

    refusal_scores = [
        result["refusal_accuracy"]
        for result in question_results
    ]

    summary = {
        "total_questions": len(
            question_results
        ),
        "successful_questions": len(
            successful_results
        ),
        "failed_questions": (
            len(question_results)
            - len(successful_results)
        ),
        "faithfulness": average(
            faithfulness_scores
        ),
        "answer_relevancy": average(
            answer_relevancy_scores
        ),
        "answer_correctness": average(
            answer_correctness_scores
        ),
        "refusal_accuracy": average(
            refusal_scores
        ),
    }

    output = {
        "version": version,
        "retrieval_k": k,
        "summary": summary,
        "questions": question_results,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{version}_generation_results.json"
    )

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
    print("Generation evaluation complete")
    print("=" * 60)

    print(
        f"Results saved to: {output_path}"
    )

    print()

    print(
        f"Faithfulness: "
        f"{summary['faithfulness']:.4f}"
    )

    print(
        f"Answer Relevancy: "
        f"{summary['answer_relevancy']:.4f}"
    )

    print(
        f"Answer Correctness: "
        f"{summary['answer_correctness']:.4f}"
    )

    print(
        f"Refusal Accuracy: "
        f"{summary['refusal_accuracy']:.4f}"
    )

    return output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Litmus RAG generation evaluation."
        )
    )

    parser.add_argument(
        "--version",
        default="v1",
        help=(
            "Evaluation version, for example v1."
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help=(
            "Number of documents to retrieve."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of questions to process."
        ),
    )

    args = parser.parse_args()

    run_generation_evaluation(
        version=args.version,
        k=args.k,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()