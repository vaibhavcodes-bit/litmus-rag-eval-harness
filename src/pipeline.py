from src.retrieval.retriever import retrieve_documents
from src.retrieval.decomposition import decompose_and_retrieve
from src.retrieval.router import route_question
from src.retrieval.sql_source import (
    count_jobs,
    average_salary,
    search_jobs,
)
from src.generation.llm_client import generate_answer


def _documents_to_sources(documents):
    """
    Convert LangChain Documents into the source format
    returned by the API.
    """

    sources = []

    for document in documents:
        sources.append(
            {
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "content": document.page_content,
            }
        )

    return sources


def _sql_result_to_context(result):
    """
    Convert structured SQL results into text context
    that can be passed to the existing LLM.
    """

    if isinstance(result, list):
        if not result:
            return ""

        lines = []

        for row in result:
            lines.append(str(row))

        return "\n".join(lines)

    if isinstance(result, dict):
        return str(result)

    return str(result)


def _run_sql_source(question: str):
    """
    Run the SQL source for supported structured job questions.

    The SQL source is intentionally explicit in V5 rather than
    generating arbitrary SQL from the user's question.
    """

    question_lower = question.lower()

    # ---------------------------------------------------------
    # Job count queries
    # ---------------------------------------------------------

    if "how many" in question_lower and "remote" in question_lower:
        result = count_jobs(remote=True)

        return {
            "context": _sql_result_to_context(
                f"Number of remote jobs: {result}"
            ),
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_remote_jobs",
                }
            ],
        }

    if "how many" in question_lower and "senior" in question_lower:
        result = count_jobs(
            experience_level="senior"
        )

        return {
            "context": _sql_result_to_context(
                f"Number of senior jobs: {result}"
            ),
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_senior_jobs",
                }
            ],
        }

    if "how many" in question_lower and (
        "entry-level" in question_lower
        or "entry level" in question_lower
    ):
        result = count_jobs(
            experience_level="entry"
        )

        return {
            "context": _sql_result_to_context(
                f"Number of entry-level jobs: {result}"
            ),
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_entry_level_jobs",
                }
            ],
        }

    if "how many" in question_lower and "bangalore" in question_lower:
        result = count_jobs(
            location="Bangalore"
        )

        return {
            "context": _sql_result_to_context(
                f"Number of jobs in Bangalore: {result}"
            ),
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_bangalore_jobs",
                }
            ],
        }
    # ---------------------------------------------------------
    # Average salary queries
    # ---------------------------------------------------------

    if "average" in question_lower and "salary" in question_lower:
        experience_level = None
        remote = None
        location = None

        if "senior" in question_lower:
            experience_level = "senior"

        if "remote" in question_lower:
            remote = True

        if "bangalore" in question_lower:
            location = "Bangalore"

        result = average_salary(
            experience_level=experience_level,
            remote=remote,
            location=location,
        )

        return {
            "context": _sql_result_to_context(
                f"Average salary data: {result}"
            ),
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "average_salary",
                }
            ],
        }

    # ---------------------------------------------------------
    # Job listing queries
    # ---------------------------------------------------------
    if "how many" in question_lower and (
        "cloudworks" in question_lower
    ):
        result = search_jobs(
            company="CloudWorks",
            limit=100,
        )

        return {
            "context": _sql_result_to_context(
                result
            ),
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_company_jobs",
                }
            ],
        }
    
    
    if (
        "which jobs" in question_lower
        or "what jobs" in question_lower
        or "jobs available" in question_lower
    ):
        location = None

        if "bangalore" in question_lower:
            location = "Bangalore"

        result = search_jobs(
            location=location,
            limit=10,
        )

        return {
            "context": _sql_result_to_context(result),
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "search_jobs",
                }
            ],
        }

    raise ValueError(
        "The SQL source does not yet support this question pattern."
    )


def answer_question(
    question: str,
    k: int = 4,
    mode: str = "v1",
):
    """
    Run the RAG pipeline.

    Supported modes:

        v1:
            Question
                ↓
            Baseline Retrieval
                ↓
            Context
                ↓
            LLM

        v4:
            Question
                ↓
            Decompose
                ↓
            Sub-question Retrieval
                ↓
            Combined Context
                ↓
            LLM

                v5:
            Question
                ↓
              Router
             /     \\
         VECTOR    SQL
           ↓        ↓
        Chroma   SQLite
             \\    /
              Context
                 ↓
                LLM
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if mode not in {"v1", "v4", "v5"}:
        raise ValueError(
            "mode must be either 'v1', 'v4', or 'v5'."
        )

    # ---------------------------------------------------------
    # V1 — Baseline RAG
    # ---------------------------------------------------------
    if mode == "v1":

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
            context_parts.append(
                document.page_content
            )

        context = "\n\n".join(context_parts)

        answer = generate_answer(
            question=question,
            context=context,
        )

        sources = _documents_to_sources(
            documents
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    # ---------------------------------------------------------
    # V4 — Query Decomposition
    # ---------------------------------------------------------
    if mode == "v4":

        result = decompose_and_retrieve(
            question=question,
            k=k,
        )

        sub_questions = result["sub_questions"]

        documents_by_sub_question = result[
            "documents_by_sub_question"
        ]

        context = result["context"]

        # Flatten documents so the API keeps a simple
        # sources[] response structure.
        all_documents = []

        for documents in documents_by_sub_question:
            all_documents.extend(documents)

        if not all_documents:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "sub_questions": sub_questions,
            }

        answer = generate_answer(
            question=question,
            context=context,
        )

        sources = _documents_to_sources(
            all_documents
        )

        return {
            "answer": answer,
            "sources": sources,
            "sub_questions": sub_questions,
        }

    # ---------------------------------------------------------
    # V5 — Multi-source Router
    # ---------------------------------------------------------
    if mode == "v5":

        source = route_question(question)

        # -----------------------------------------------------
        # VECTOR source
        # -----------------------------------------------------
        if source == "VECTOR":

            documents = retrieve_documents(
                question=question,
                k=k,
            )

            if not documents:
                return {
                    "answer": "I don't have enough information to answer that.",
                    "sources": [],
                    "route": "VECTOR",
                }

            context_parts = []

            for document in documents:
                context_parts.append(
                    document.page_content
                )

            context = "\n\n".join(context_parts)

            answer = generate_answer(
                question=question,
                context=context,
            )

            sources = _documents_to_sources(
                documents
            )

            return {
                "answer": answer,
                "sources": sources,
                "route": "VECTOR",
            }

        # -----------------------------------------------------
        # SQL source
        # -----------------------------------------------------
        if source == "SQL":

            result = _run_sql_source(question)

            context = result["context"]
            sources = result["sources"]

            if not context:
                return {
                    "answer": "I don't have enough information to answer that.",
                    "sources": sources,
                    "route": "SQL",
                }

            answer = generate_answer(
                question=question,
                context=context,
            )

            return {
                "answer": answer,
                "sources": sources,
                "route": "SQL",
            }

        raise ValueError(
            f"Unsupported route returned by router: {source}"
        )