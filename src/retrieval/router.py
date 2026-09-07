import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"

SourceType = Literal["VECTOR", "SQL"]


def get_router_llm() -> ChatGroq:
    """
    Create the LLM used to classify questions into a data source.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set in the environment."
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )


def _clean_router_output(raw_output: str) -> str:
    """
    Normalize the LLM's classification output.
    """

    text = raw_output.strip().upper()

    # Remove common formatting such as:
    # "VECTOR"
    # "**VECTOR**"
    # "SOURCE: VECTOR"
    text = text.replace("**", "")
    text = text.replace("`", "")

    match = re.search(r"\b(VECTOR|SQL)\b", text)

    if not match:
        raise ValueError(
            f"Router returned an invalid source: {raw_output!r}"
        )

    return match.group(1)


def route_question(question: str) -> SourceType:
    """
    Route a question to the most appropriate source.

    Returns:
        "VECTOR" for unstructured document questions.
        "SQL" for structured database questions.
    """

    if not question or not question.strip():
        raise ValueError("question must not be empty")

    llm = get_router_llm()

    prompt = f"""
You are a routing classifier for a Retrieval-Augmented Generation system.

The system has two data sources:

1. VECTOR
   Use VECTOR when the question asks for information contained
   in unstructured documents such as resume guides, ATS guidance,
   cover-letter guidance, interview preparation material, LinkedIn
   guidance, salary negotiation guidance, or career advice.

2. SQL
   Use SQL when the question asks for structured job-listing data,
   counts, filtering, aggregation, salaries, locations, companies,
   remote status, experience levels, or other fields that exist
   in a jobs database.

Examples:

Question: What should a resume summary contain?
Answer: VECTOR

Question: What makes a cover letter effective?
Answer: VECTOR

Question: How many remote jobs are available?
Answer: SQL

Question: How many senior jobs are available?
Answer: SQL

Question: What is the average maximum salary for senior roles?
Answer: SQL

Question: Which jobs are available in Bangalore?
Answer: SQL

Return ONLY one word:
VECTOR
or
SQL

Do not explain your answer.

Question:
{question}
"""

    response = llm.invoke(prompt)

    raw_output = response.content

    if isinstance(raw_output, list):
        raw_output = " ".join(
            str(item)
            for item in raw_output
        )

    return _clean_router_output(str(raw_output))


if __name__ == "__main__":
    test_questions = [
        "What should a resume summary contain?",
        "How many remote jobs are available?",
        "What makes a cover letter effective?",
        "How many senior jobs are available?",
        "What is the average maximum salary for senior roles?",
    ]

    for question in test_questions:
        source = route_question(question)

        print(f"Question: {question}")
        print(f"Route: {source}")
        print()