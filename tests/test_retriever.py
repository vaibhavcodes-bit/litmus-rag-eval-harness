from src.retrieval.retriever import retrieve_documents


def test_retrieve_documents():
    question = "What are the company working hours?"

    documents = retrieve_documents(question, k=2)

    assert len(documents) > 0

    first_document = documents[0]

    assert first_document.page_content.strip() != ""
    assert "source" in first_document.metadata
    assert "page" in first_document.metadata


def test_retrieve_documents_rejects_empty_question():
    try:
        retrieve_documents("")
        assert False, "Expected ValueError"
    except ValueError:
        pass