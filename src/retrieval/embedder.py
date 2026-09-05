from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embeddings():
    """
    Create the local embedding model used by V1.

    Model:
    sentence-transformers/all-MiniLM-L6-v2
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings