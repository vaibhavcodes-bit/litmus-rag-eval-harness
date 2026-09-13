from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings():
    """
    Load the embedding model once and reuse it.

    The LRU cache prevents the HuggingFace embedding model
    from being recreated for every request.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings