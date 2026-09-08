from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.generation.llm_client import generate_answer
from src.retrieval.answer_grader import grade_answer
from src.retrieval.grader import (
    MAX_RETRIES,
    grade_documents,
    rewrite_query,
)
from src.retrieval.retriever import retrieve_documents
from src.retrieval.router import route_question
from src.retrieval.sql_source import (
    average_salary,
    count_jobs,
    search_jobs,
)


MAX_ANSWER_RETRIES = 2


class RAGState(TypedDict, total=False):
    question: str

    route: Literal["VECTOR", "SQL", "WEB"]

    documents: list[Any]
    graded_documents: list[Any]
    context: str

    rewritten_queries: list[str]

    sql_result: Any

    answer: str
    answer_grade: Literal["GOOD", "BAD"]

    sources: list[dict]

    retrieval_attempts: int
    answer_attempts: int
    relevant: bool


# ============================================================
# ROUTE
# ============================================================

def route_node(state: RAGState) -> dict:
    question = state["question"]

    print(f"[V7] Route node: {question}")

    route = route_question(question)

    print(f"[V7] Route decision: {route}")

    return {
        "route": route,
        "retrieval_attempts": state.get("retrieval_attempts", 0),
        "answer_attempts": state.get("answer_attempts", 0),
        "rewritten_queries": state.get("rewritten_queries", []),
    }


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def vector_retrieve_node(state: RAGState) -> dict:
    question = state["question"]

    retrieval_attempts = state.get("retrieval_attempts", 0) + 1

    print(
        f"[V7] VECTOR retrieve node "
        f"(attempt {retrieval_attempts})"
    )

    # If this is a retry, rewrite the question first.
    rewritten_queries = state.get("rewritten_queries", [])

    retrieval_question = question

    if retrieval_attempts > 1:
        rewritten_query = rewrite_query(question)

        rewritten_queries = [
            *rewritten_queries,
            rewritten_query,
        ]

        retrieval_question = rewritten_query

        print(
            f"[V7] Rewritten query: "
            f"{rewritten_query}"
        )

    documents = retrieve_documents(
        retrieval_question,
        k=4,
    )

    return {
        "documents": documents,
        "retrieval_attempts": retrieval_attempts,
        "rewritten_queries": rewritten_queries,
    }


# ============================================================
# GRADE DOCUMENTS
# ============================================================

def grade_documents_node(state: RAGState) -> dict:
    question = state["question"]
    documents = state.get("documents", [])

    print("[V7] Grade documents node")

    graded_documents = grade_documents(
        question,
        documents,
    )

    relevant_documents = [
        document
        for document, grade in graded_documents
        if grade == "RELEVANT"
    ]

    print(
        f"[V7] Relevant documents: "
        f"{len(relevant_documents)}/{len(documents)}"
    )

    return {
        "graded_documents": graded_documents,
        "relevant": len(relevant_documents) > 0,
    }


# ============================================================
# BUILD VECTOR CONTEXT
# ============================================================

def build_vector_context_node(state: RAGState) -> dict:
    graded_documents = state.get("graded_documents", [])

    relevant_documents = [
        document
        for document, grade in graded_documents
        if grade == "RELEVANT"
    ]

    if not relevant_documents:
        return {
            "context": "",
            "sources": [],
        }

    context_parts = []
    sources = []

    for document in relevant_documents:
        content = document.page_content

        context_parts.append(content)

        sources.append(
            {
                "source": document.metadata.get(
                    "source",
                    "unknown",
                ),
                "page": document.metadata.get(
                    "page",
                    0,
                ),
                "content": content,
            }
        )

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }


# ============================================================
# VECTOR GENERATION
# ============================================================

def vector_generate_node(state: RAGState) -> dict:
    question = state["question"]
    context = state.get("context", "")

    answer_attempts = state.get("answer_attempts", 0) + 1

    print(
        f"[V7] VECTOR generate node "
        f"(attempt {answer_attempts})"
    )

    if not context:
        answer = (
            "I don't have enough information "
            "to answer that question."
        )
    else:
        answer = generate_answer(
            question,
            context,
        )

    return {
        "answer": answer,
        "answer_attempts": answer_attempts,
    }


# ============================================================
# SQL
# ============================================================

def sql_node(state: RAGState) -> dict:
    question = state["question"]

    print("[V7] SQL node")

    question_lower = question.lower()

    # --------------------------------------------------------
    # Count remote jobs
    # --------------------------------------------------------

    if "how many" in question_lower and "remote" in question_lower:
        result = count_jobs(remote=True)

        print(f"[V7] SQL result: {result}")

        return {
            "sql_result": result,
            "context": f"Number of remote jobs: {result}",
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_remote_jobs",
                }
            ],
        }

    # --------------------------------------------------------
    # Count senior jobs
    # --------------------------------------------------------

    if "how many" in question_lower and "senior" in question_lower:
        result = count_jobs(
            experience_level="Senior",
        )

        print(f"[V7] SQL result: {result}")

        return {
            "sql_result": result,
            "context": f"Number of senior jobs: {result}",
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_senior_jobs",
                }
            ],
        }

    # --------------------------------------------------------
    # Count entry-level jobs
    # --------------------------------------------------------

    if (
        "how many" in question_lower
        and (
            "entry" in question_lower
            or "entry-level" in question_lower
        )
    ):
        result = count_jobs(
            experience_level="Entry",
        )

        print(f"[V7] SQL result: {result}")

        return {
            "sql_result": result,
            "context": f"Number of entry-level jobs: {result}",
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_entry_jobs",
                }
            ],
        }

    # --------------------------------------------------------
    # Bangalore jobs
    # --------------------------------------------------------

    if (
        "how many" in question_lower
        and (
            "bangalore" in question_lower
            or "bengaluru" in question_lower
        )
    ):
        result = count_jobs(
            location="Bangalore",
        )

        print(f"[V7] SQL result: {result}")

        return {
            "sql_result": result,
            "context": f"Number of Bangalore jobs: {result}",
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "count_bangalore_jobs",
                }
            ],
        }

    # --------------------------------------------------------
    # Average salary
    # --------------------------------------------------------

    if (
        "average salary" in question_lower
        or "average compensation" in question_lower
    ):
        result = average_salary()

        context = (
            f"Average salary: {result}"
        )

        print(f"[V7] SQL result: {result}")

        return {
            "sql_result": result,
            "context": context,
            "sources": [
                {
                    "source": "jobs.db",
                    "type": "sql",
                    "query_type": "average_salary",
                }
            ],
        }

    # --------------------------------------------------------
    # General SQL job search
    # --------------------------------------------------------

    results = search_jobs(
        limit=10,
    )

    print(
        f"[V7] SQL result: "
        f"{len(results)} jobs"
    )

    if not results:
        return {
            "sql_result": [],
            "context": "",
            "sources": [],
        }

    context_lines = []

    for job in results:
        context_lines.append(
            (
                f"Job ID: {job.get('job_id')}\n"
                f"Title: {job.get('title')}\n"
                f"Company: {job.get('company')}\n"
                f"Location: {job.get('location')}\n"
                f"Remote: {job.get('remote')}\n"
                f"Experience: {job.get('experience_level')}\n"
                f"Salary Min: {job.get('salary_min')}\n"
                f"Salary Max: {job.get('salary_max')}"
            )
        )

    return {
        "sql_result": results,
        "context": "\n\n".join(context_lines),
        "sources": [
            {
                "source": "jobs.db",
                "type": "sql",
                "query_type": "search_jobs",
            }
        ],
    }


# ============================================================
# SQL GENERATION
# ============================================================

def sql_generate_node(state: RAGState) -> dict:
    question = state["question"]
    context = state.get("context", "")

    answer_attempts = state.get("answer_attempts", 0) + 1

    print(
        f"[V7] SQL generate node "
        f"(attempt {answer_attempts})"
    )

    if not context:
        answer = (
            "I don't have enough information "
            "to answer that question."
        )
    else:
        answer = generate_answer(
            question,
            context,
        )

    return {
        "answer": answer,
        "answer_attempts": answer_attempts,
    }


# ============================================================
# ANSWER GRADING
# ============================================================

def grade_answer_node(state: RAGState) -> dict:
    question = state["question"]
    context = state.get("context", "")
    answer = state.get("answer", "")

    print("[V7] Grade answer node")

    if not context or not answer:
        grade = "BAD"
    else:
        grade = grade_answer(
            question,
            context,
            answer,
        )

    print(f"[V7] Answer grade: {grade}")

    return {
        "answer_grade": grade,
    }


# ============================================================
# RETRY
# ============================================================

def retry_node(state: RAGState) -> dict:
    print("[V7] Retry node")

    # Clear data from the previous retrieval/generation
    # so the next iteration produces a fresh result.
    return {
        "documents": [],
        "graded_documents": [],
        "context": "",
        "sources": [],
        "answer": "",
        "answer_grade": "BAD",
        "relevant": False,
    }


# ============================================================
# RETRY BRANCH
# ============================================================

def retry_branch(
    state: RAGState,
) -> Literal["vector", "sql"]:
    """
    Preserve the original source route during retries.

    VECTOR questions retry through vector retrieval.
    SQL questions retry through SQL.
    """

    route = state.get("route")

    if route == "SQL":
        return "sql"

    return "vector"


# ============================================================
# ROUTING BRANCH
# ============================================================

def route_branch(
    state: RAGState,
) -> Literal["vector", "sql"]:
    route = state.get("route")

    if route == "SQL":
        return "sql"

    return "vector"


# ============================================================
# ANSWER-GRADE BRANCH
# ============================================================

def answer_grade_branch(
    state: RAGState,
) -> Literal["end", "retry"]:
    grade = state.get(
        "answer_grade",
        "BAD",
    )

    answer_attempts = state.get(
        "answer_attempts",
        0,
    )

    if grade == "GOOD":
        return "end"

    # Total allowed answer attempts:
    # 1 initial attempt + MAX_ANSWER_RETRIES retries
    if answer_attempts >= MAX_ANSWER_RETRIES + 1:
        print(
            "[V7] Maximum answer retries reached"
        )

        return "end"

    print(
        "[V7] Answer grade BAD -> retry"
    )

    return "retry"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():
    graph = StateGraph(RAGState)

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    graph.add_node(
        "route",
        route_node,
    )

    graph.add_node(
        "vector_retrieve",
        vector_retrieve_node,
    )

    graph.add_node(
        "grade_documents",
        grade_documents_node,
    )

    graph.add_node(
        "build_vector_context",
        build_vector_context_node,
    )

    graph.add_node(
        "vector_generate",
        vector_generate_node,
    )

    graph.add_node(
        "sql",
        sql_node,
    )

    graph.add_node(
        "sql_generate",
        sql_generate_node,
    )

    graph.add_node(
        "grade_answer",
        grade_answer_node,
    )

    graph.add_node(
        "retry",
        retry_node,
    )

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "route",
    )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "route",
        route_branch,
        {
            "vector": "vector_retrieve",
            "sql": "sql",
        },
    )

    # --------------------------------------------------------
    # VECTOR PATH
    # --------------------------------------------------------

    graph.add_edge(
        "vector_retrieve",
        "grade_documents",
    )

    graph.add_edge(
        "grade_documents",
        "build_vector_context",
    )

    graph.add_edge(
        "build_vector_context",
        "vector_generate",
    )

    graph.add_edge(
        "vector_generate",
        "grade_answer",
    )

    # --------------------------------------------------------
    # SQL PATH
    # --------------------------------------------------------

    graph.add_edge(
        "sql",
        "sql_generate",
    )

    graph.add_edge(
        "sql_generate",
        "grade_answer",
    )

    # --------------------------------------------------------
    # ANSWER GRADING
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "grade_answer",
        answer_grade_branch,
        {
            "end": END,
            "retry": "retry",
        },
    )

    # --------------------------------------------------------
    # RETRY
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Retry must preserve the original route.
    #
    # VECTOR -> vector_retrieve
    # SQL    -> sql
    #

    graph.add_conditional_edges(
        "retry",
        retry_branch,
        {
            "vector": "vector_retrieve",
            "sql": "sql",
        },
    )

    return graph.compile()


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    question = (
        "How many remote jobs are available?"
    )

    app = build_graph()

    final_state = app.invoke(
        {
            "question": question,
        }
    )

    print("\n[V7] Final state:")
    print(final_state)