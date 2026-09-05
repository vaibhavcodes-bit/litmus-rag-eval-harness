from src.retrieval.vector_store import get_vector_store


def get_retriever(k: int = 4):
    """
    Create a retriever that returns the top-k
    most relevant chunks from Chroma.
    """

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )

    return retriever


def retrieve_documents(question: str, k: int = 4):
    """
    Retrieve the most relevant document chunks
    for a user question.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    retriever = get_retriever(k=k)

    documents = retriever.invoke(question)

    return documents