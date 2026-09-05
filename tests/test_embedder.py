from src.retrieval.embedder import get_embeddings


def test_get_embeddings():
    embeddings = get_embeddings()

    result = embeddings.embed_query(
        "What are the company working hours?"
    )

    assert result
    assert len(result) > 0

    assert all(isinstance(value, float) for value in result)