from typing import Any


def hit_rate_at_k(
    retrieved_documents: list[dict[str, Any]],
    expected_sources: list[str],
    k: int = 4,
) -> float:
    """
    Return 1.0 if any expected source is found
    in the top-k retrieved documents, otherwise 0.0.
    """

    if not expected_sources:
        return 0.0

    top_k_documents = retrieved_documents[:k]

    retrieved_sources = {
        document.get("source")
        for document in top_k_documents
    }

    return float(
        any(
            source in retrieved_sources
            for source in expected_sources
        )
    )


def mean_reciprocal_rank(
    retrieved_documents: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank for one question.

    Reciprocal rank:
        1 / rank of the first relevant document

    If no relevant document is found:
        0.0
    """

    if not expected_sources:
        return 0.0

    for rank, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        source = document.get("source")

        if source in expected_sources:
            return 1.0 / rank

    return 0.0


def context_precision(
    retrieved_documents: list[dict[str, Any]],
    expected_sources: list[str],
) -> float:
    """
    Calculate the percentage of retrieved documents
    that come from expected sources.
    """

    if not retrieved_documents:
        return 0.0

    relevant_count = 0

    for document in retrieved_documents:
        if document.get("source") in expected_sources:
            relevant_count += 1

    return relevant_count / len(retrieved_documents)


def refusal_accuracy(
    answer: str,
    should_refuse: bool,
) -> float:
    """
    Check whether the system correctly refuses
    an unanswerable question.

    This is intentionally simple for V2.
    """

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
    ]

    refused = any(
        phrase in answer_lower
        for phrase in refusal_phrases
    )

    if should_refuse:
        return float(refused)

    return float(not refused)


def average(values: list[float]) -> float:
    """
    Calculate the arithmetic mean.
    """

    if not values:
        return 0.0

    return sum(values) / len(values)