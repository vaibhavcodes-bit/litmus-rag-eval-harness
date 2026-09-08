from langchain_core.documents import Document

from src.retrieval.grader import corrective_retrieve


def test_corrective_retrieve_succeeds_on_first_attempt(monkeypatch):
    documents = [
        Document(
            page_content="Behavioral interview preparation and STAR method."
        )
    ]

    calls = []

    def fake_retrieve(query, k):
        calls.append((query, k))
        return documents

    monkeypatch.setattr(
        "src.retrieval.grader.grade_documents",
        lambda question, documents: [
            (documents[0], "RELEVANT")
        ],
    )

    result = corrective_retrieve(
        question="How should I prepare for a behavioral interview?",
        retrieve_fn=fake_retrieve,
        k=4,
    )

    assert result["relevant"] is True
    assert result["retrieval_attempts"] == 1
    assert result["rewritten_queries"] == []
    assert len(result["documents"]) == 1
    assert calls == [
        ("How should I prepare for a behavioral interview?", 4)
    ]


def test_corrective_retrieve_rewrites_after_bad_retrieval(monkeypatch):
    first_document = Document(
        page_content="Salary negotiation strategies."
    )

    second_document = Document(
        page_content="Behavioral interview preparation and STAR method."
    )

    calls = []

    def fake_retrieve(query, k):
        calls.append((query, k))

        if len(calls) == 1:
            return [first_document]

        return [second_document]

    grades = iter(
        [
            [(first_document, "NOT_RELEVANT")],
            [(second_document, "RELEVANT")],
        ]
    )

    monkeypatch.setattr(
        "src.retrieval.grader.grade_documents",
        lambda question, documents: next(grades),
    )

    monkeypatch.setattr(
        "src.retrieval.grader.rewrite_query",
        lambda question: "behavioral interview preparation STAR method",
    )

    result = corrective_retrieve(
        question="How should I prepare for a behavioral interview?",
        retrieve_fn=fake_retrieve,
        k=4,
    )

    assert result["relevant"] is True
    assert result["retrieval_attempts"] == 2
    assert result["rewritten_queries"] == [
        "behavioral interview preparation STAR method"
    ]
    assert result["documents"] == [second_document]

    assert calls == [
        ("How should I prepare for a behavioral interview?", 4),
        ("behavioral interview preparation STAR method", 4),
    ]


def test_corrective_retrieve_stops_after_max_two_retries(monkeypatch):
    document = Document(
        page_content="Completely unrelated information."
    )

    calls = []

    def fake_retrieve(query, k):
        calls.append((query, k))
        return [document]

    monkeypatch.setattr(
        "src.retrieval.grader.grade_documents",
        lambda question, documents: [
            (document, "NOT_RELEVANT")
        ],
    )

    rewrite_count = []

    def fake_rewrite(question):
        rewrite_count.append(question)
        return f"rewritten query {len(rewrite_count)}"

    monkeypatch.setattr(
        "src.retrieval.grader.rewrite_query",
        fake_rewrite,
    )

    result = corrective_retrieve(
        question="Some question",
        retrieve_fn=fake_retrieve,
        k=4,
        max_retries=2,
    )

    assert result["relevant"] is False

    # Initial attempt + 2 retries = 3 retrieval attempts.
    assert result["retrieval_attempts"] == 3

    # Exactly two query rewrites.
    assert len(result["rewritten_queries"]) == 2

    # Retrieval was called exactly three times.
    assert len(calls) == 3


def test_corrective_retrieve_rejects_empty_question():
    def fake_retrieve(query, k):
        return []

    try:
        corrective_retrieve(
            question="",
            retrieve_fn=fake_retrieve,
        )
        assert False
    except ValueError as error:
        assert str(error) == "question must not be empty."


def test_corrective_retrieve_rejects_invalid_k():
    def fake_retrieve(query, k):
        return []

    try:
        corrective_retrieve(
            question="test question",
            retrieve_fn=fake_retrieve,
            k=0,
        )
        assert False
    except ValueError as error:
        assert str(error) == "k must be greater than 0."


def test_corrective_retrieve_rejects_negative_retries():
    def fake_retrieve(query, k):
        return []

    try:
        corrective_retrieve(
            question="test question",
            retrieve_fn=fake_retrieve,
            max_retries=-1,
        )
        assert False
    except ValueError as error:
        assert (
            str(error)
            == "max_retries must be greater than or equal to 0."
        )