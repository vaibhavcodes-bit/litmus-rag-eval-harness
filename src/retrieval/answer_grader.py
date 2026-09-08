import os
from typing import Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"


AnswerGrade = Literal["GOOD", "BAD"]


def get_answer_grader_llm() -> ChatGroq:
    """
    Create the LLM used to judge whether a generated answer
    is supported by the supplied context and answers the question.
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


def _clean_answer_grade(raw_output: str) -> AnswerGrade:
    """
    Convert the LLM output into a strict GOOD / BAD value.
    """

    text = raw_output.strip().upper()

    text = text.replace("**", "")
    text = text.replace("`", "")

    if "BAD" in text:
        return "BAD"

    if "GOOD" in text:
        return "GOOD"

    raise ValueError(
        f"Answer grader returned an invalid grade: {raw_output!r}"
    )


def grade_answer(
    question: str,
    context: str,
    answer: str,
) -> AnswerGrade:
    """
    Judge whether the generated answer correctly answers
    the user's question using only the supplied context.
    """

    if not question or not question.strip():
        raise ValueError(
            "question must not be empty."
        )

    if not context or not context.strip():
        raise ValueError(
            "context must not be empty."
        )

    if not answer or not answer.strip():
        raise ValueError(
            "answer must not be empty."
        )

    llm = get_answer_grader_llm()

    prompt = f"""
You are an answer-quality grader for a
Retrieval-Augmented Generation system.

Your job is to determine whether the generated answer:

1. Directly answers the user's question.
2. Is supported by the supplied context.
3. Does not invent facts that are not supported by the context.

Return:

GOOD

if the answer correctly answers the question and is supported
by the context.

Return:

BAD

if the answer is incorrect, incomplete in a materially important way,
unsupported by the context, or contains invented information.

Do not rewrite the answer.
Do not answer the user's question yourself.
Only judge the generated answer.

Return ONLY one label:
GOOD
or
BAD

USER QUESTION:
{question}

CONTEXT:
{context}

GENERATED ANSWER:
{answer}
"""

    response = llm.invoke(prompt)

    raw_output = response.content

    if isinstance(raw_output, list):
        raw_output = " ".join(
            str(item)
            for item in raw_output
        )

    return _clean_answer_grade(
        str(raw_output)
    )


if __name__ == "__main__":

    question = (
        "What should a resume summary contain?"
    )

    context = """
A resume summary highlights existing experience
and achievements and suits experienced candidates.
"""

    answer = (
    "A resume summary should contain "
    "your home address, passport number, "
    "and bank account details."
)

    grade = grade_answer(
        question=question,
        context=context,
        answer=answer,
    )

    print("Answer grade:", grade)