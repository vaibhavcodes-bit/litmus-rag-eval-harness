from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset

from ragas import evaluate
from ragas.llms import llm_factory

from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET_PATH = (
    PROJECT_ROOT
    / "eval"
    / "golden_dataset.json"
)

CACHE_PATH = (
    PROJECT_ROOT
    / "eval"
    / "results"
    / "v1_generation_cache.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "eval"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "v2_generation_results.json"
)

PROGRESS_PATH = (
    RESULTS_DIR
    / "v2_ragas_progress.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)


# ============================================================
# CONTROLLED RAGAS EVALUATION SETTINGS
# ============================================================

# Default token budget for Ragas metrics.
MAX_TOKENS = 2048

# AnswerCorrectness produces a more complex structured
# evaluation response, so give it additional output space.
CORRECTNESS_MAX_TOKENS = 2048

# Retry each individual metric.
RETRY_COUNT = 3

# Wait before retrying a failed metric.
RETRY_DELAY_SECONDS = 15

# Wait between individual Ragas metric calls.
METRIC_DELAY_SECONDS = 5

# Wait between questions.
QUESTION_DELAY_SECONDS = 10
# ============================================================
# JSON HELPERS
# ============================================================

def load_json(path: Path) -> Any:

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# REFUSAL ACCURACY
# ============================================================

def refusal_accuracy(
    answer: str,
    should_refuse: bool,
) -> float:

    if not answer:

        return 0.0

    answer_lower = answer.lower()

    refusal_phrases = [
        "i don't have enough information",
        "i do not have enough information",
        "not enough information",
        "cannot answer",
        "can't answer",
        "no relevant information",
        "i'm unable to answer",
        "i am unable to answer",
        "outside the provided context",
        "outside the available context",
    ]

    refused = any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )

    if should_refuse:

        return float(refused)

    return float(not refused)


# ============================================================
# BUILD RAGAS LLM
# ============================================================

def build_ragas_llm(
    max_tokens: int = MAX_TOKENS,
):
    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found "
            "in the .env file."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=(
            "https://api.groq.com/openai/v1"
        ),
    )

    ragas_llm = llm_factory(
        model=MODEL_NAME,
        provider="openai",
        client=client,
        max_tokens=max_tokens,
    )

    return ragas_llm

# ============================================================
# BUILD LOCAL EMBEDDINGS
# ============================================================

def build_ragas_embeddings():

    print(
        "Loading local Ragas embeddings..."
    )

    local_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    ragas_embeddings = (
        LangchainEmbeddingsWrapper(
            local_embeddings
        )
    )

    print(
        "Ragas embeddings: "
        f"{EMBEDDING_MODEL}"
    )

    return ragas_embeddings


# ============================================================
# LOAD GOLDEN QUESTIONS
# ============================================================

def load_questions():

    golden_data = load_json(
        DATASET_PATH
    )

    if isinstance(
        golden_data,
        dict,
    ):

        questions = golden_data.get(
            "questions",
            [],
        )

    else:

        questions = golden_data

    return questions


# ============================================================
# LOAD V1 GENERATION CACHE
# ============================================================

def load_generation_cache():

    cache_data = load_json(
        CACHE_PATH
    )

    return cache_data.get(
        "questions",
        [],
    )


# ============================================================
# BUILD QUESTION MAP
# ============================================================

def build_cache_map(
    cache_questions: list[dict[str, Any]],
):

    return {
        item["id"]: item
        for item in cache_questions
    }


# ============================================================
# BUILD SINGLE RAGAS DATASET
# ============================================================

def build_single_dataset(
    question: dict[str, Any],
    cached: dict[str, Any],
):

    answer = cached.get(
        "answer",
        "",
    )

    contexts = cached.get(
        "retrieved_contexts",
        [],
    )

    ground_truth = question.get(
        "ground_truth",
        cached.get(
            "ground_truth",
            "",
        ),
    )

    dataset_row = {
        "question": question["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
    }

    return Dataset.from_list(
        [dataset_row]
    )


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value):

    if value is None:

        return None

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None

    if value != value:

        return None

    return value


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================
def evaluate_question(
    question: dict[str, Any],
    cached: dict[str, Any],
    ragas_llm,
    correctness_llm,
    ragas_embeddings,
):

    question_id = question["id"]

    answer = cached.get(
        "answer",
        "",
    )

    contexts = cached.get(
        "retrieved_contexts",
        [],
    )

    should_refuse = bool(
        question.get(
            "should_refuse",
            False,
        )
    )

    if not answer:

        raise ValueError(
            f"{question_id} has no cached answer."
        )

    if not contexts:

        raise ValueError(
            f"{question_id} has no retrieved contexts."
        )

    dataset = build_single_dataset(
        question,
        cached,
    )

    # ========================================================
    # CONTROLLED METRIC DEFINITIONS
    # ========================================================

    metric_definitions = [
        (
            "faithfulness",
            Faithfulness(
                llm=ragas_llm
            ),
        ),
        (
            "answer_relevancy",
            AnswerRelevancy(
                llm=ragas_llm
            ),
        ),
        (
            "answer_correctness",
            AnswerCorrectness(
                llm=correctness_llm
            ),
        ),
    ]

    metric_results = {
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
    }

    errors = []

    # ========================================================
    # CONTROLLED SEQUENTIAL METRIC EVALUATION
    # ========================================================

    for metric_name, metric in metric_definitions:

        metric_success = False
        last_error = None

        for attempt in range(
            1,
            RETRY_COUNT + 1,
        ):

            try:

                print(
                    f"  {metric_name}: "
                    f"attempt {attempt}/"
                    f"{RETRY_COUNT}"
                )

                # ------------------------------------------------
                # Evaluate ONLY ONE metric at a time.
                # ------------------------------------------------

                result = evaluate(
                    dataset=dataset,
                    metrics=[metric],
                    embeddings=ragas_embeddings,
                )

                result_df = result.to_pandas()

                if result_df.empty:

                    raise RuntimeError(
                        f"Ragas returned an empty "
                        f"result for {metric_name}."
                    )

                row = result_df.iloc[0]

                value = safe_float(
                    row.get(metric_name)
                )

                if value is None:

                    raise RuntimeError(
                        f"Ragas returned no valid "
                        f"value for {metric_name}."
                    )

                metric_results[
                    metric_name
                ] = value

                print(
                    f"  {metric_name}: {value}"
                )

                metric_success = True

                # ------------------------------------------------
                # CONTROLLED DELAY
                # ------------------------------------------------

                if (
                    metric_name
                    != "answer_correctness"
                ):

                    print(
                        f"  Waiting "
                        f"{METRIC_DELAY_SECONDS} "
                        f"seconds before next metric..."
                    )

                    time.sleep(
                        METRIC_DELAY_SECONDS
                    )

                break

            except Exception as error:

                last_error = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                print(
                    f"  {metric_name} failed: "
                    f"{last_error}"
                )

                if attempt < RETRY_COUNT:

                    print(
                        f"  Waiting "
                        f"{RETRY_DELAY_SECONDS} "
                        f"seconds before retry..."
                    )

                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

        # --------------------------------------------------------
        # IMPORTANT CHANGE:
        #
        # If one metric fails after all retries, DO NOT stop
        # the entire question.
        #
        # Keep that metric as None and continue evaluating
        # the remaining metrics.
        # --------------------------------------------------------

        if not metric_success:

            errors.append(
                f"{metric_name}: "
                f"{last_error}"
            )

            print(
                f"  {question_id}: "
                f"{metric_name} could not "
                f"be evaluated."
            )

            print(
                f"  Continuing with the next metric..."
            )

            # NO break here.
            continue

    # ========================================================
    # DETERMINE EVALUATION STATUS
    # ========================================================

    missing_metrics = [
        name
        for name, value
        in metric_results.items()
        if value is None
    ]

    refusal_score = refusal_accuracy(
        answer,
        should_refuse,
    )

    # ========================================================
    # PARTIAL RESULT
    # ========================================================

    if missing_metrics:

        error_message = (
            "One or more Ragas metrics could not "
            "be evaluated. Missing metrics: "
            + ", ".join(missing_metrics)
        )

        if errors:

            error_message += (
                " | "
                + " | ".join(errors)
            )

        return {
            "id": question_id,
            "question": question["question"],
            "faithfulness": metric_results[
                "faithfulness"
            ],
            "answer_relevancy": metric_results[
                "answer_relevancy"
            ],
            "answer_correctness": metric_results[
                "answer_correctness"
            ],
            "refusal_accuracy": refusal_score,
            "should_refuse": should_refuse,
            "status": "partial",
            "error": error_message,
        }

    # ========================================================
    # COMPLETE SUCCESS
    # ========================================================

    return {
        "id": question_id,
        "question": question["question"],
        "faithfulness": metric_results[
            "faithfulness"
        ],
        "answer_relevancy": metric_results[
            "answer_relevancy"
        ],
        "answer_correctness": metric_results[
            "answer_correctness"
        ],
        "refusal_accuracy": refusal_score,
        "should_refuse": should_refuse,
        "status": "success",
        "error": None,
    }

# ============================================================
# LOAD PROGRESS
# ============================================================

def load_progress():

    if not PROGRESS_PATH.exists():

        return {
            "version": "v2",
            "model": MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "questions": {},
        }

    return load_json(
        PROGRESS_PATH
    )


# ============================================================
# SAVE PROGRESS
# ============================================================

def save_progress(
    progress: dict[str, Any],
):

    save_json(
        PROGRESS_PATH,
        progress,
    )


# ============================================================
# CALCULATE AVERAGE
# ============================================================

def calculate_average(
    question_results: list[dict[str, Any]],
    metric_name: str,
):

    values = []

    for result in question_results:

        value = result.get(
            metric_name
        )

        if value is None:

            continue

        try:

            values.append(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

    if not values:

        return None

    return sum(values) / len(values)


# ============================================================
# BUILD FINAL RESULTS
# ============================================================

def build_final_output(
    progress: dict[str, Any],
):

    question_results = list(
        progress[
            "questions"
        ].values()
    )

    question_results.sort(
        key=lambda item: item["id"]
    )

    # ========================================================
    # CLASSIFY RESULTS
    # ========================================================

    successful_results = [
        item
        for item in question_results
        if item.get("status")
        == "success"
    ]

    partial_results = [
        item
        for item in question_results
        if item.get("status")
        == "partial"
    ]

    failed_results = [
        item
        for item in question_results
        if item.get("status")
        == "failed"
    ]

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "total_questions": len(
            question_results
        ),

        "successful_questions": len(
            successful_results
        ),

        "partial_questions": len(
            partial_results
        ),

        "failed_questions": len(
            failed_results
        ),

        "faithfulness": (
            calculate_average(
                question_results,
                "faithfulness",
            )
        ),

        "answer_relevancy": (
            calculate_average(
                question_results,
                "answer_relevancy",
            )
        ),

        "answer_correctness": (
            calculate_average(
                question_results,
                "answer_correctness",
            )
        ),

        "refusal_accuracy": (
            calculate_average(
                question_results,
                "refusal_accuracy",
            )
        ),
    }

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    return {
        "version": "v2",

        "model": MODEL_NAME,

        "embedding_model": EMBEDDING_MODEL,

        "summary": summary,

        "questions": question_results,
    }

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Controlled, sequential and "
            "resumable Ragas evaluation "
            "for Litmus V2."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N "
            "questions."
        ),
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry questions that "
            "previously failed."
        ),
    )

    args = parser.parse_args()

    print("=" * 60)

    print(
        "Litmus V2 - Controlled Sequential "
        "Ragas Evaluation"
    )

    print("=" * 60)

    # ========================================================
    # LOAD QUESTIONS
    # ========================================================

    questions = load_questions()

    print(
        f"Questions in golden dataset: "
        f"{len(questions)}"
    )

    # ========================================================
    # LOAD GENERATION CACHE
    # ========================================================

    cache_questions = (
        load_generation_cache()
    )

    print(
        f"Cached generation results: "
        f"{len(cache_questions)}"
    )

    cache_by_id = build_cache_map(
        cache_questions
    )

    # ========================================================
    # SELECT QUESTIONS
    # ========================================================

    if args.limit is not None:

        selected_questions = (
            questions[:args.limit]
        )

    else:

        selected_questions = questions

    print(
        f"Questions selected: "
        f"{len(selected_questions)}"
    )

    # ========================================================
    # LOAD PROGRESS
    # ========================================================

    progress = load_progress()

    previous_results = progress.get(
        "questions",
        {},
    )

    print(
        f"Previously evaluated: "
        f"{len(previous_results)}"
    )

    # ========================================================
    # DETERMINE PENDING QUESTIONS
    # ========================================================

    pending_questions = []

    for question in selected_questions:

        question_id = question["id"]

        previous = previous_results.get(
            question_id
        )

        if previous is None:

            pending_questions.append(
                question
            )

            continue

        if (
            previous.get("status")
            == "success"
        ):

            print(
                f"Skipping {question_id} "
                f"(already successful)"
            )

            continue

        if args.retry_failed:

            print(
                f"Retrying {question_id} "
                f"(previously failed)"
            )

            pending_questions.append(
                question
            )

        else:

            print(
                f"Skipping {question_id} "
                f"(previously failed; "
                f"use --retry-failed)"
            )

    print(
        f"Questions to evaluate now: "
        f"{len(pending_questions)}"
    )

    if not pending_questions:

        print()

        print(
            "No new questions need evaluation."
        )

        final_output = build_final_output(
            progress
        )

        save_json(
            OUTPUT_PATH,
            final_output,
        )

        print(
            f"Results saved to: "
            f"{OUTPUT_PATH}"
        )

        return

    # ========================================================
    # VALIDATE CACHE
    # ========================================================

    usable_questions = []

    for question in pending_questions:

        question_id = question["id"]

        cached = cache_by_id.get(
            question_id
        )

        if cached is None:

            print(
                f"WARNING: No generation "
                f"cache for {question_id}"
            )

            progress[
                "questions"
            ][question_id] = {
                "id": question_id,
                "question": question[
                    "question"
                ],
                "faithfulness": None,
                "answer_relevancy": None,
                "answer_correctness": None,
                "refusal_accuracy": None,
                "should_refuse": bool(
                    question.get(
                        "should_refuse",
                        False,
                    )
                ),
                "status": "failed",
                "error": (
                    "No generation cache found."
                ),
            }

            save_progress(
                progress
            )

            continue

        if not cached.get("answer"):

            print(
                f"WARNING: Empty answer "
                f"for {question_id}"
            )

            continue

        if not cached.get(
            "retrieved_contexts"
        ):

            print(
                f"WARNING: Empty contexts "
                f"for {question_id}"
            )

            continue

        usable_questions.append(
            question
        )

    print(
        f"Usable questions to evaluate: "
        f"{len(usable_questions)}"
    )

    if not usable_questions:

        print(
            "No usable questions remain."
        )

        return

    # ========================================================
    # CREATE EVALUATOR
    # ========================================================

    print()

    print(
        "Creating Ragas evaluator..."
    )

    ragas_llm = build_ragas_llm(
    max_tokens=MAX_TOKENS
)

    correctness_llm = build_ragas_llm(
        max_tokens=CORRECTNESS_MAX_TOKENS
    )

    print(
    f"Standard metric max tokens: "
    f"{MAX_TOKENS}"
)

    print(
        f"AnswerCorrectness max tokens: "
        f"{CORRECTNESS_MAX_TOKENS}"
    )

    ragas_embeddings = (
        build_ragas_embeddings()
    )

    # ========================================================
    # CONTROLLED SEQUENTIAL EVALUATION
    # ========================================================

    total = len(
        usable_questions
    )

    print()

    print(
        "Starting CONTROLLED sequential "
        "evaluation..."
    )

    print(
        f"Total questions in this run: "
        f"{total}"
    )

    print(
        f"Metrics are evaluated ONE AT A TIME."
    )

    print(
        f"Metric delay: "
        f"{METRIC_DELAY_SECONDS} seconds"
    )

    print(
        f"Question delay: "
        f"{QUESTION_DELAY_SECONDS} seconds"
    )

    print()

    for index, question in enumerate(
        usable_questions,
        start=1,
    ):

        question_id = question["id"]

        print("=" * 60)

        print(
            f"[{index}/{total}] "
            f"Evaluating {question_id}"
        )

        print(
            f"Question: "
            f"{question['question']}"
        )

        cached = cache_by_id[
            question_id
        ]

        try:

           result = evaluate_question(
                question=question,
                cached=cached,
                ragas_llm=ragas_llm,
                correctness_llm=correctness_llm,
                ragas_embeddings=(
                    ragas_embeddings
                ),
            )

        except Exception as error:

            result = {
                "id": question_id,
                "question": question[
                    "question"
                ],
                "faithfulness": None,
                "answer_relevancy": None,
                "answer_correctness": None,
                "refusal_accuracy": (
                    refusal_accuracy(
                        cached.get(
                            "answer",
                            "",
                        ),
                        bool(
                            question.get(
                                "should_refuse",
                                False,
                            )
                        ),
                    )
                ),
                "should_refuse": bool(
                    question.get(
                        "should_refuse",
                        False,
                    )
                ),
                "status": "failed",
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

        # ====================================================
        # SAVE IMMEDIATELY
        # ====================================================

        progress[
            "questions"
        ][question_id] = result

        save_progress(
            progress
        )

        print()

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Faithfulness: "
            f"{result['faithfulness']}"
        )

        print(
            f"Answer Relevancy: "
            f"{result['answer_relevancy']}"
        )

        print(
            f"Answer Correctness: "
            f"{result['answer_correctness']}"
        )

        print(
            f"Refusal Accuracy: "
            f"{result['refusal_accuracy']}"
        )

        if result.get("error"):

            print(
                f"Error: "
                f"{result['error']}"
            )

        print(
            "Progress saved."
        )

        # ====================================================
        # CONTROLLED DELAY BETWEEN QUESTIONS
        # ====================================================

        if index < total:

            print()

            print(
                f"Waiting "
                f"{QUESTION_DELAY_SECONDS} "
                f"seconds before next question..."
            )

            time.sleep(
                QUESTION_DELAY_SECONDS
            )

            print()

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    final_output = build_final_output(
        progress
    )

    save_json(
        OUTPUT_PATH,
        final_output,
    )

    summary = final_output[
        "summary"
    ]

    print()

    print("=" * 60)

    print(
        "V2 Controlled Ragas Evaluation Complete"
    )

    print("=" * 60)

    print(
        f"Total questions: "
        f"{summary['total_questions']}"
    )

    print(
        f"Successful questions: "
        f"{summary['successful_questions']}"
    )

    print(
        f"Failed questions: "
        f"{summary['failed_questions']}"
    )

    print()

    print(
        f"Faithfulness: "
        f"{summary['faithfulness']}"
    )

    print(
        f"Answer Relevancy: "
        f"{summary['answer_relevancy']}"
    )

    print(
        f"Answer Correctness: "
        f"{summary['answer_correctness']}"
    )

    print(
        f"Refusal Accuracy: "
        f"{summary['refusal_accuracy']}"
    )

    print()

    print(
        "Progress saved to:"
    )

    print(
        PROGRESS_PATH
    )

    print()

    print(
        "Final results saved to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()