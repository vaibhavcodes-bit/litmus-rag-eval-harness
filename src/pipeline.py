from src.retrieval.retriever import retrieve_documents
from src.generation.llm_client import generate_answer


def answer_question(question: str, k: int = 4):
    """
    Run the complete V1 RAG pipeline.

    Flow:
        Question
            ↓
        Retriever
            ↓
        Relevant documents
            ↓
        Context
            ↓
        LLM
            ↓
        Answer + Sources
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    documents = retrieve_documents(
        question=question,
        k=k,
    )

    if not documents:
        return {
            "answer": "I don't have enough information to answer that.",
            "sources": [],
        }

    context_parts = []

    for document in documents:
        context_parts.append(document.page_content)

    context = "\n\n".join(context_parts)

    answer = generate_answer(
        question=question,
        context=context,
    )

    sources = []

    for document in documents:
        sources.append(
            {
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "content": document.page_content,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }