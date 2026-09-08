from langchain_core.documents import Document

from src.retrieval.grader import corrective_retrieve


def test_v6_corrective_retrieval_realistic_retry():
    """
    Deterministic V6 benchmark:

    Attempt 1 intentionally returns an irrelevant document.
    The rewritten query causes attempt 2 to return the relevant document.

    This verifies:
    - document grading
    - query rewriting
    - retry behavior
    - corrective success
    """

    relevant_document = Document(
        page_content=(
            "First step: Begin with self-assessment: clarify career goals, "
            "target roles, target industries, and non-negotiables before applying."
        ),
        metadata={
            "source": "job_search_strategy.pdf",
            "page": 0,
        },
    )

    irrelevant_document = Document(
        page_content=(
            "Salary negotiation should consider market benchmarks, "
            "total compensation, and competing offers."
        ),
        metadata={
            "source": "salary_negotiation.pdf",
            "page": 0,
        },
    )

    calls = []

    def controlled_retriever(query, k=4):
        calls.append(query)

        if len(calls) == 1:
            return [irrelevant_document]

        return [relevant_document]

    result = corrective_retrieve(
        question="What is the recommended first step before starting a job search?",
        retrieve_fn=controlled_retriever,
        k=4,
        max_retries=2,
    )

    assert result["relevant"] is True
    assert result["retrieval_attempts"] == 2
    assert len(result["rewritten_queries"]) == 1
    assert result["rewritten_queries"][0].strip()

    assert len(result["documents"]) == 1
    assert result["documents"][0].metadata["source"] == "job_search_strategy.pdf"

    assert calls[0] != calls[1]