import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

Grade = Literal["RELEVANT", "NOT_RELEVANT"]


def get_grader_llm() -> ChatGroq:
    """
    Create the LLM used to judge whether a retrieved document
    is relevant to the user's question.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment.")

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )


def _clean_grade_output(raw_output: str) -> Grade:
    """
    Convert the LLM output into a strict RELEVANT / NOT_RELEVANT value.
    """

    text = raw_output.strip().upper()

    text = text.replace("**", "")
    text = text.replace("`", "")

    if "NOT_RELEVANT" in text:
        return "NOT_RELEVANT"

    if "RELEVANT" in text:
        return "RELEVANT"

    raise ValueError(
        f"Grader returned an invalid grade: {raw_output!r}"
    )


def grade_document(
    question: str,
    document: Document,
) -> Grade:
    """
    Grade one retrieved document against the user's question.
    """

    if not question or not question.strip():
        raise ValueError("question must not be empty.")

    if document is None:
        raise ValueError("document must not be None.")

    content = document.page_content.strip()

    if not content:
        raise ValueError("document content must not be empty.")

    llm = get_grader_llm()

    prompt = f"""
You are a document relevance grader for a Retrieval-Augmented
Generation system.

Your job is to determine whether the document contains information
that is useful for answering the user's question.

Return:

RELEVANT

if the document contains information that directly helps answer
the question.

Return:

NOT_RELEVANT

if the document does not contain useful information for answering
the question.

Do not judge whether the document is well written.
Do not answer the user's question.
Only judge document relevance.

Return ONLY one label:
RELEVANT
or
NOT_RELEVANT

USER QUESTION:
{question}

DOCUMENT:
{content}
"""

    response = llm.invoke(prompt)

    raw_output = response.content

    if isinstance(raw_output, list):
        raw_output = " ".join(str(item) for item in raw_output)

    return _clean_grade_output(str(raw_output))


def grade_documents(
    question: str,
    documents: list[Document],
) -> list[tuple[Document, Grade]]:
    """
    Grade all retrieved documents.

    Returns:
        A list of (document, grade) tuples.
    """

    if not question or not question.strip():
        raise ValueError("question must not be empty.")

    if documents is None:
        raise ValueError("documents must not be None.")

    results = []

    for document in documents:
        grade = grade_document(
            question=question,
            document=document,
        )

        results.append((document, grade))

    return results


def filter_relevant_documents(
    graded_documents: list[tuple[Document, Grade]],
) -> list[Document]:
    """
    Keep only documents graded as RELEVANT.
    """

    if graded_documents is None:
        raise ValueError("graded_documents must not be None.")

    return [
        document
        for document, grade in graded_documents
        if grade == "RELEVANT"
    ]


def has_relevant_documents(
    graded_documents: list[tuple[Document, Grade]],
) -> bool:
    """
    Return True if at least one document is relevant.
    """

    if graded_documents is None:
        raise ValueError("graded_documents must not be None.")

    return any(
        grade == "RELEVANT"
        for _, grade in graded_documents
    )
    
    
    
def rewrite_query(question):
    """
    Rewrite a failed retrieval query into a materially different,
    keyword-rich search query.
    """

    if not question or not question.strip():
        raise ValueError("question must not be empty")

    llm = get_grader_llm()

    prompt = f"""
You are a search-query rewriting expert for a career and resume knowledge base.

The original user question is:

{question}

The previous retrieval attempt failed to find the correct document.

Create ONE materially different search query that is more likely to retrieve
the correct knowledge-base chunk.

Rules:
- Do NOT simply shorten or repeat the original question.
- Use likely domain terminology and concepts that the answer/document would contain.
- Replace conversational wording with specific searchable concepts.
- Focus on the information the user is actually asking for.
- Keep the query concise.
- Output ONLY the rewritten search query.
- Do not explain your reasoning.

Example:

Question:
"What is the recommended first step before starting a job search?"

Bad rewrite:
"recommended first step before starting a job search"

Good rewrite:
"job search preparation self-assessment career goals target roles target industries"

Now rewrite this question:

{question}
"""

    response = llm.invoke(prompt)

    rewritten = response.content.strip()

    # Remove accidental surrounding quotes.
    rewritten = rewritten.strip('"').strip("'").strip()

    # Remove common conversational prefixes.
    rewritten = re.sub(
        r"^(rewritten query|search query|query)\s*:\s*",
        "",
        rewritten,
        flags=re.IGNORECASE,
    ).strip()

    if not rewritten:
        return question.strip()

    return rewritten


MAX_RETRIES = 2


def corrective_retrieve(
    question: str,
    retrieve_fn,
    k: int = 4,
    max_retries: int = MAX_RETRIES,
):
    """
    Retrieve documents, grade them, and rewrite the query when
    none of the retrieved documents are relevant.

    Returns:
        {
            "documents": [...],
            "graded_documents": [...],
            "query": "...",
            "rewritten_queries": [...],
            "retrieval_attempts": N,
            "relevant": True/False,
        }
    """

    if not question or not question.strip():
        raise ValueError("question must not be empty.")

    if retrieve_fn is None:
        raise ValueError("retrieve_fn must not be None.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if max_retries < 0:
        raise ValueError("max_retries must be greater than or equal to 0.")

    current_query = question.strip()
    rewritten_queries = []
    retrieval_attempts = 0

    best_graded_documents = []

    for attempt in range(max_retries + 1):
        documents = retrieve_fn(current_query, k)
        retrieval_attempts += 1

        graded_documents = grade_documents(
            question=question,
            documents=documents,
        )

        best_graded_documents = graded_documents

        if has_relevant_documents(graded_documents):
            relevant_documents = filter_relevant_documents(
                graded_documents
            )

            return {
                "documents": relevant_documents,
                "graded_documents": graded_documents,
                "query": current_query,
                "rewritten_queries": rewritten_queries,
                "retrieval_attempts": retrieval_attempts,
                "relevant": True,
            }

        if attempt == max_retries:
            break

        current_query = rewrite_query(question)
        rewritten_queries.append(current_query)

    return {
        "documents": filter_relevant_documents(
            best_graded_documents
        ),
        "graded_documents": best_graded_documents,
        "query": current_query,
        "rewritten_queries": rewritten_queries,
        "retrieval_attempts": retrieval_attempts,
        "relevant": False,
    }