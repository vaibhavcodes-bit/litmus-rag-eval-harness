from langchain_core.documents import Document

from src.retrieval.grader import (
    _clean_grade_output,
    filter_relevant_documents,
    grade_document,
    grade_documents,
    has_relevant_documents,
)


def test_clean_grade_output_relevant():
    assert _clean_grade_output("RELEVANT") == "RELEVANT"


def test_clean_grade_output_not_relevant():
    assert _clean_grade_output("NOT_RELEVANT") == "NOT_RELEVANT"


def test_clean_grade_output_markdown():
    assert _clean_grade_output("**RELEVANT**") == "RELEVANT"


def test_clean_grade_output_with_extra_text():
    assert (
        _clean_grade_output(
            "The document is useful.\nRELEVANT"
        )
        == "RELEVANT"
    )


def test_clean_grade_output_invalid():
    try:
        _clean_grade_output("UNKNOWN")
        assert False
    except ValueError as error:
        assert "invalid grade" in str(error)


def test_grade_document_relevant(monkeypatch):
    class FakeResponse:
        content = "RELEVANT"

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr(
        "src.retrieval.grader.get_grader_llm",
        lambda: FakeLLM(),
    )

    document = Document(
        page_content="Behavioral interviews use the STAR method."
    )

    result = grade_document(
        "How should I prepare for a behavioral interview?",
        document,
    )

    assert result == "RELEVANT"


def test_grade_document_not_relevant(monkeypatch):
    class FakeResponse:
        content = "NOT_RELEVANT"

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr(
        "src.retrieval.grader.get_grader_llm",
        lambda: FakeLLM(),
    )

    document = Document(
        page_content="Salary negotiation includes compensation bands."
    )

    result = grade_document(
        "How should I prepare for a behavioral interview?",
        document,
    )

    assert result == "NOT_RELEVANT"


def test_grade_document_rejects_empty_question():
    document = Document(
        page_content="Behavioral interviews use the STAR method."
    )

    try:
        grade_document("", document)
        assert False
    except ValueError as error:
        assert str(error) == "question must not be empty."


def test_grade_document_rejects_empty_document(monkeypatch):
    class FakeLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM should not be called.")

    monkeypatch.setattr(
        "src.retrieval.grader.get_grader_llm",
        lambda: FakeLLM(),
    )

    document = Document(page_content="")

    try:
        grade_document(
            "How should I prepare for a behavioral interview?",
            document,
        )
        assert False
    except ValueError as error:
        assert str(error) == "document content must not be empty."


def test_grade_documents(monkeypatch):
    grades = iter(
        [
            "RELEVANT",
            "NOT_RELEVANT",
        ]
    )

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse(next(grades))

    monkeypatch.setattr(
        "src.retrieval.grader.get_grader_llm",
        lambda: FakeLLM(),
    )

    documents = [
        Document(page_content="Interview preparation and STAR method."),
        Document(page_content="Salary negotiation strategies."),
    ]

    result = grade_documents(
        "How should I prepare for a behavioral interview?",
        documents,
    )

    assert len(result) == 2
    assert result[0][1] == "RELEVANT"
    assert result[1][1] == "NOT_RELEVANT"


def test_filter_relevant_documents():
    relevant_document = Document(
        page_content="Behavioral interview preparation."
    )

    irrelevant_document = Document(
        page_content="Salary negotiation."
    )

    graded_documents = [
        (relevant_document, "RELEVANT"),
        (irrelevant_document, "NOT_RELEVANT"),
    ]

    result = filter_relevant_documents(graded_documents)

    assert result == [relevant_document]


def test_has_relevant_documents_true():
    document = Document(
        page_content="Behavioral interview preparation."
    )

    graded_documents = [
        (document, "RELEVANT"),
    ]

    assert has_relevant_documents(graded_documents) is True


def test_has_relevant_documents_false():
    document = Document(
        page_content="Salary negotiation."
    )

    graded_documents = [
        (document, "NOT_RELEVANT"),
    ]

    assert has_relevant_documents(graded_documents) is False