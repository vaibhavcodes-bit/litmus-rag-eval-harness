from eval.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    context_precision,
    refusal_accuracy,
    average,
)


def test_hit_rate_at_k_when_relevant_document_is_found():
    documents = [
        {"source": "wrong.pdf"},
        {"source": "resume_writing.pdf"},
        {"source": "another.pdf"},
    ]

    result = hit_rate_at_k(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
        k=3,
    )

    assert result == 1.0


def test_hit_rate_at_k_when_relevant_document_is_not_found():
    documents = [
        {"source": "wrong.pdf"},
        {"source": "another.pdf"},
    ]

    result = hit_rate_at_k(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
        k=2,
    )

    assert result == 0.0


def test_hit_rate_at_k_respects_k():
    documents = [
        {"source": "wrong.pdf"},
        {"source": "another.pdf"},
        {"source": "resume_writing.pdf"},
    ]

    result = hit_rate_at_k(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
        k=2,
    )

    assert result == 0.0


def test_mean_reciprocal_rank():
    documents = [
        {"source": "wrong.pdf"},
        {"source": "resume_writing.pdf"},
        {"source": "another.pdf"},
    ]

    result = mean_reciprocal_rank(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
    )

    assert result == 0.5


def test_mean_reciprocal_rank_when_first_document_is_correct():
    documents = [
        {"source": "resume_writing.pdf"},
        {"source": "wrong.pdf"},
    ]

    result = mean_reciprocal_rank(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
    )

    assert result == 1.0


def test_mean_reciprocal_rank_when_document_is_not_found():
    documents = [
        {"source": "wrong.pdf"},
        {"source": "another.pdf"},
    ]

    result = mean_reciprocal_rank(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
    )

    assert result == 0.0


def test_context_precision():
    documents = [
        {"source": "resume_writing.pdf"},
        {"source": "wrong.pdf"},
        {"source": "resume_writing.pdf"},
        {"source": "another.pdf"},
    ]

    result = context_precision(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
    )

    assert result == 0.5


def test_context_precision_when_all_documents_are_relevant():
    documents = [
        {"source": "resume_writing.pdf"},
        {"source": "resume_writing.pdf"},
    ]

    result = context_precision(
        retrieved_documents=documents,
        expected_sources=["resume_writing.pdf"],
    )

    assert result == 1.0


def test_refusal_accuracy_when_system_should_refuse():
    answer = "I don't have enough information to answer that."

    result = refusal_accuracy(
        answer=answer,
        should_refuse=True,
    )

    assert result == 1.0


def test_refusal_accuracy_when_system_should_not_refuse():
    answer = "Use measurable achievements in your resume."

    result = refusal_accuracy(
        answer=answer,
        should_refuse=False,
    )

    assert result == 1.0


def test_refusal_accuracy_when_system_should_refuse_but_does_not():
    answer = "Employees receive 20 days of annual leave."

    result = refusal_accuracy(
        answer=answer,
        should_refuse=True,
    )

    assert result == 0.0


def test_average():
    result = average([1.0, 0.5, 0.0])

    assert result == 0.5


def test_average_empty_list():
    result = average([])

    assert result == 0.0