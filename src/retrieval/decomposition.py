import os
import re
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.retrieval.retriever import retrieve_documents


load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"

DEFAULT_RETRIEVAL_K = 4
DEFAULT_MAX_SUB_QUESTIONS = 4


def get_decomposition_llm():
    """
    Create the LLM used to decompose a user question
    into independent sub-questions.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to the .env file in the project root."
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )


def decompose_question(
    question: str,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
) -> List[str]:
    """
    Decompose a question into independent sub-questions.

    Simple questions should return one sub-question.
    Compound or comparison questions should return multiple
    independent sub-questions.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if max_sub_questions < 1:
        raise ValueError(
            "max_sub_questions must be at least 1."
        )

    prompt = f"""
You are a question decomposition system for a document-based RAG system.

Your task is to break the user's question into independent
sub-questions that can be retrieved separately from a vector database.

IMPORTANT RULES:

- If the question asks for only one piece of information,
  return exactly one question.
- If the question contains multiple independent information needs,
  split it into separate questions.
- For comparison questions, create one sub-question for each
  thing being compared.
- Each sub-question must be understandable on its own.
- Preserve the important subject and terminology from the original question.
- Do not answer the questions.
- Do not add explanations.
- Output ONLY the sub-questions.
- One sub-question per line.
- Do not number the questions.
- Do not use bullet points.
- Return at most {max_sub_questions} sub-questions.

Examples:

Input:
Compare the leave policy and work-from-home policy.

Output:
What is the leave policy?
What is the work-from-home policy?

Input:
What is the company's leave policy?

Output:
What is the company's leave policy?

Input:
Compare resume summaries and resume objectives.

Output:
What is a resume summary?
What is a resume objective?

User question:
{question}
"""

    llm = get_decomposition_llm()

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = " ".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    sub_questions = []

    for line in str(content).splitlines():
        sub_question = line.strip()

        if not sub_question:
            continue

        # Remove bullets and numbering if the model adds them.
        sub_question = re.sub(
            r"^[-*•]\s*",
            "",
            sub_question,
        )

        sub_question = re.sub(
            r"^\d+[\.\)]\s*",
            "",
            sub_question,
        )

        sub_question = sub_question.strip()

        if not sub_question:
            continue

        # Ignore conversational model responses.
        lower_question = sub_question.lower()

        if (
            "could you please provide" in lower_question
            or "please provide the question" in lower_question
            or "i need the question" in lower_question
            or "what is the question" in lower_question
        ):
            continue

        # Remove duplicate sub-questions.
        if sub_question.lower() not in {
            existing.lower()
            for existing in sub_questions
        }:
            sub_questions.append(sub_question)

    # Safety fallback:
    # if decomposition fails, retrieve using the original question.
    if not sub_questions:
        sub_questions.append(question.strip())

    return sub_questions[:max_sub_questions]


def retrieve_for_sub_questions(
    sub_questions: List[str],
    k: int = DEFAULT_RETRIEVAL_K,
) -> List[List[Document]]:
    """
    Retrieve documents independently for every sub-question.
    """

    if not sub_questions:
        raise ValueError(
            "sub_questions must not be empty."
        )

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    results = []

    for sub_question in sub_questions:
        documents = retrieve_documents(
            question=sub_question,
            k=k,
        )

        results.append(documents)

    return results


def combine_subquestion_contexts(
    sub_questions: List[str],
    documents_by_sub_question: List[List[Document]],
) -> str:
    """
    Combine retrieved documents while preserving the relationship
    between each sub-question and its retrieved context.
    """

    if len(sub_questions) != len(
        documents_by_sub_question
    ):
        raise ValueError(
            "Number of sub-questions must match "
            "number of document groups."
        )

    sections = []

    for index, (
        sub_question,
        documents,
    ) in enumerate(
        zip(
            sub_questions,
            documents_by_sub_question,
        ),
        start=1,
    ):
        sections.append(
            f"SUB-QUESTION {index}:\n"
            f"{sub_question}\n\n"
            f"RETRIEVED CONTEXT:\n"
        )

        if not documents:
            sections.append(
                "No relevant documents were retrieved."
            )
            continue

        for document_index, document in enumerate(
            documents,
            start=1,
        ):
            source = document.metadata.get(
                "source",
                "unknown",
            )

            sections.append(
                f"[Document {document_index} | "
                f"Source: {source}]\n"
                f"{document.page_content.strip()}\n"
            )

    return "\n".join(sections)


def decompose_and_retrieve(
    question: str,
    max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
    k: int = DEFAULT_RETRIEVAL_K,
):
    """
    Complete V4 retrieval flow.

    1. Decompose the original question.
    2. Retrieve independently for every sub-question.
    3. Combine the retrieved contexts with labels.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    sub_questions = decompose_question(
        question=question,
        max_sub_questions=max_sub_questions,
    )

    documents_by_sub_question = retrieve_for_sub_questions(
        sub_questions=sub_questions,
        k=k,
    )

    combined_context = combine_subquestion_contexts(
        sub_questions=sub_questions,
        documents_by_sub_question=documents_by_sub_question,
    )

    return {
        "sub_questions": sub_questions,
        "documents_by_sub_question": documents_by_sub_question,
        "context": combined_context,
    }