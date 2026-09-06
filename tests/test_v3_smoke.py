from src.retrieval.multi_query import multi_query_retrieve


def test_v3_multi_query_retrieval_returns_relevant_document():
    question = "What is the recommended first step before starting a job search?"

    documents = multi_query_retrieve(
        question=question,
        query_count=4,
        k=4,
    )

    assert documents
    assert len(documents) <= 16

    combined_text = "\n".join(
        document.page_content.lower()
        for document in documents
    )

    assert "self-assessment" in combined_text
    assert "career goals" in combined_text