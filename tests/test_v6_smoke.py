from langchain_core.documents import Document

from src.pipeline import answer_question


def test_v6_vector_relevant_on_first_attempt(monkeypatch):
    document = Document(
        page_content="Behavioral interviews often use the STAR method.",
        metadata={
            "source": "interview_prep.pdf",
            "page": 1,
        },
    )

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "VECTOR",
    )

    monkeypatch.setattr(
        "src.pipeline.corrective_retrieve",
        lambda question, retrieve_fn, k, max_retries: {
            "documents": [document],
            "graded_documents": [
                (document, "RELEVANT")
            ],
            "query": question,
            "rewritten_queries": [],
            "retrieval_attempts": 1,
            "relevant": True,
        },
    )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        lambda question, context: "Use the STAR method.",
    )

    result = answer_question(
        "How should I prepare for a behavioral interview?",
        mode="v6",
    )

    assert result["route"] == "VECTOR"
    assert result["answer"] == "Use the STAR method."
    assert result["retrieval_attempts"] == 1
    assert result["rewritten_queries"] == []
    assert result["relevant"] is True

    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "interview_prep.pdf"


def test_v6_vector_rewrites_after_irrelevant_retrieval(monkeypatch):
    first_document = Document(
        page_content="Salary negotiation strategies.",
        metadata={
            "source": "salary_negotiation.pdf",
            "page": 1,
        },
    )

    second_document = Document(
        page_content="Behavioral interview preparation and STAR method.",
        metadata={
            "source": "interview_prep.pdf",
            "page": 2,
        },
    )

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "VECTOR",
    )

    monkeypatch.setattr(
        "src.pipeline.corrective_retrieve",
        lambda question, retrieve_fn, k, max_retries: {
            "documents": [second_document],
            "graded_documents": [
                (second_document, "RELEVANT")
            ],
            "query": "behavioral interview preparation STAR method",
            "rewritten_queries": [
                "behavioral interview preparation STAR method"
            ],
            "retrieval_attempts": 2,
            "relevant": True,
        },
    )

    captured = {}

    def fake_generate_answer(question, context):
        captured["question"] = question
        captured["context"] = context
        return "Prepare using the STAR method."

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        fake_generate_answer,
    )

    result = answer_question(
        "How should I prepare for a behavioral interview?",
        mode="v6",
    )

    assert result["route"] == "VECTOR"
    assert result["answer"] == "Prepare using the STAR method."

    assert result["retrieval_attempts"] == 2

    assert result["rewritten_queries"] == [
        "behavioral interview preparation STAR method"
    ]

    assert result["relevant"] is True

    assert (
        "Behavioral interview preparation and STAR method."
        in captured["context"]
    )

    assert result["sources"][0]["source"] == "interview_prep.pdf"


def test_v6_vector_returns_refusal_after_max_retries(monkeypatch):
    irrelevant_document = Document(
        page_content="Completely unrelated salary information.",
        metadata={
            "source": "salary_negotiation.pdf",
            "page": 1,
        },
    )

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "VECTOR",
    )

    monkeypatch.setattr(
        "src.pipeline.corrective_retrieve",
        lambda question, retrieve_fn, k, max_retries: {
            "documents": [],
            "graded_documents": [
                (irrelevant_document, "NOT_RELEVANT")
            ],
            "query": "rewritten query 2",
            "rewritten_queries": [
                "rewritten query 1",
                "rewritten query 2",
            ],
            "retrieval_attempts": 3,
            "relevant": False,
        },
    )

    result = answer_question(
        "Some question with no relevant information.",
        mode="v6",
    )

    assert result["route"] == "VECTOR"

    assert (
        result["answer"]
        == "I don't have enough information to answer that."
    )

    assert result["sources"] == []

    assert result["retrieval_attempts"] == 3

    assert len(result["rewritten_queries"]) == 2

    assert result["relevant"] is False


def test_v6_sql_route_preserves_v5_behavior(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "SQL",
    )

    monkeypatch.setattr(
        "src.pipeline.count_jobs",
        lambda remote=True: 14,
    )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        lambda question, context: "There are 14 remote jobs.",
    )

    result = answer_question(
        "How many remote jobs are available?",
        mode="v6",
    )

    assert result["route"] == "SQL"
    assert result["answer"] == "There are 14 remote jobs."

    assert result["sources"][0]["source"] == "jobs.db"
    assert result["sources"][0]["type"] == "sql"

    assert result["retrieval_attempts"] == 1
    assert result["rewritten_queries"] == []
    assert result["relevant"] is True


def test_v6_invalid_mode():
    try:
        answer_question(
            "Hello",
            mode="v9",
        )

        assert False

    except ValueError as error:
        assert (
            str(error)
            == "mode must be one of 'v1', 'v3', 'v4', 'v5', or 'v6'."
        )