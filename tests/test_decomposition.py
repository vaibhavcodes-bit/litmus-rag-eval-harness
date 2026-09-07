from langchain_core.documents import Document

from src.retrieval.decomposition import (
    combine_subquestion_contexts,
    decompose_question,
    retrieve_for_sub_questions,
)


def test_decompose_question_returns_multiple_for_comparison(
    monkeypatch,
):
    class FakeResponse:
        content = (
            "What is the leave policy?\n"
            "What is the work-from-home policy?"
        )

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr(
        "src.retrieval.decomposition.get_decomposition_llm",
        lambda: FakeLLM(),
    )

    result = decompose_question(
        "Compare the leave policy and work-from-home policy."
    )

    assert result == [
        "What is the leave policy?",
        "What is the work-from-home policy?",
    ]


def test_decompose_question_returns_one_for_simple_question(
    monkeypatch,
):
    class FakeResponse:
        content = "What is the leave policy?"

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr(
        "src.retrieval.decomposition.get_decomposition_llm",
        lambda: FakeLLM(),
    )

    result = decompose_question(
        "What is the leave policy?"
    )

    assert result == [
        "What is the leave policy?"
    ]


def test_decompose_question_removes_duplicate_questions(
    monkeypatch,
):
    class FakeResponse:
        content = (
            "What is the leave policy?\n"
            "What is the leave policy?\n"
            "What is the WFH policy?"
        )

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr(
        "src.retrieval.decomposition.get_decomposition_llm",
        lambda: FakeLLM(),
    )

    result = decompose_question(
        "Compare leave and WFH policies."
    )

    assert result == [
        "What is the leave policy?",
        "What is the WFH policy?",
    ]


def test_decompose_question_rejects_empty_question():
    try:
        decompose_question("")
        assert False
    except ValueError:
        assert True


def test_combine_subquestion_contexts():
    sub_questions = [
        "What is the leave policy?",
        "What is the work-from-home policy?",
    ]

    documents_by_sub_question = [
        [
            Document(
                page_content="Employees receive 20 days of leave.",
                metadata={"source": "leave_policy.pdf"},
            )
        ],
        [
            Document(
                page_content=(
                    "Employees may work remotely two days per week."
                ),
                metadata={"source": "wfh_policy.pdf"},
            )
        ],
    ]

    result = combine_subquestion_contexts(
        sub_questions,
        documents_by_sub_question,
    )

    assert "SUB-QUESTION 1" in result
    assert "SUB-QUESTION 2" in result

    assert "leave policy" in result.lower()
    assert "work-from-home policy" in result.lower()

    assert "20 days of leave" in result
    assert "two days per week" in result


def test_retrieve_for_sub_questions(
    monkeypatch,
):
    leave_document = Document(
        page_content="Leave policy information.",
        metadata={"source": "leave_policy.pdf"},
    )

    wfh_document = Document(
        page_content="Work from home information.",
        metadata={"source": "wfh_policy.pdf"},
    )

    def fake_retrieve_documents(
        question,
        k,
    ):
        if "leave" in question.lower():
            return [leave_document]

        if "work-from-home" in question.lower():
            return [wfh_document]

        return []

    monkeypatch.setattr(
        "src.retrieval.decomposition.retrieve_documents",
        fake_retrieve_documents,
    )

    sub_questions = [
        "What is the leave policy?",
        "What is the work-from-home policy?",
    ]

    results = retrieve_for_sub_questions(
        sub_questions,
        k=4,
    )

    assert len(results) == 2

    assert results[0][0].page_content == (
        "Leave policy information."
    )

    assert results[1][0].page_content == (
        "Work from home information."
    )



def test_decompose_and_retrieve(monkeypatch):
    from src.retrieval.decomposition import (
        decompose_and_retrieve,
    )

    class FakeResponse:
        content = (
            "What is the leave policy?\n"
            "What is the work-from-home policy?"
        )

    class FakeLLM:
        def invoke(self, prompt):
            return FakeResponse()

    monkeypatch.setattr(
        "src.retrieval.decomposition.get_decomposition_llm",
        lambda: FakeLLM(),
    )

    leave_document = Document(
        page_content="Employees receive 20 days of leave.",
        metadata={"source": "leave_policy.pdf"},
    )

    wfh_document = Document(
        page_content=(
            "Employees may work remotely two days per week."
        ),
        metadata={"source": "wfh_policy.pdf"},
    )

    def fake_retrieve_documents(question, k):
        if "leave" in question.lower():
            return [leave_document]

        if "work-from-home" in question.lower():
            return [wfh_document]

        return []

    monkeypatch.setattr(
        "src.retrieval.decomposition.retrieve_documents",
        fake_retrieve_documents,
    )

    result = decompose_and_retrieve(
        "Compare leave and work-from-home policies."
    )

    assert len(result["sub_questions"]) == 2

    assert len(
        result["documents_by_sub_question"]
    ) == 2

    assert "20 days of leave" in result["context"]

    assert "two days per week" in result["context"]

    assert "SUB-QUESTION 1" in result["context"]

    assert "SUB-QUESTION 2" in result["context"]
