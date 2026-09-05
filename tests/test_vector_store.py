from pathlib import Path

from langchain_core.documents import Document

from src.retrieval.vector_store import (
    get_vector_store,
    add_documents,
)


def test_vector_store():
    test_directory = Path("data/processed/chroma")

    documents = [
        Document(
            page_content="Company working hours are from 9 AM to 6 PM.",
            metadata={
                "source": "test_policy.pdf",
                "page": 0,
            },
        ),
        Document(
            page_content="Employees receive 20 annual leave days.",
            metadata={
                "source": "test_leave_policy.pdf",
                "page": 0,
            },
        ),
    ]

    vector_store = add_documents(documents)

    assert vector_store is not None

    results = vector_store.similarity_search(
        "What are the company working hours?",
        k=1,
    )

    assert len(results) == 1

    result = results[0]

    assert "working hours" in result.page_content.lower()
    assert result.metadata["source"] == "test_policy.pdf"

    assert test_directory.exists()