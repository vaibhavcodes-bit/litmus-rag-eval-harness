import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():
    """
    Create the Groq LLM client.

    Required environment variable:
    GROQ_API_KEY
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to the .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    )

    return llm


def build_prompt(question: str, context: str) -> str:
    """
    Build a grounded RAG prompt.

    The model must answer only from the supplied context.
    """

    return f"""
You are a helpful company policy assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."

Do not use outside knowledge.
Do not invent or assume facts.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def generate_answer(question: str, context: str):
    """
    Generate an answer using the Groq LLM.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        raise ValueError("Context cannot be empty.")

    llm = get_llm()

    prompt = build_prompt(
        question=question,
        context=context,
    )

    response = llm.invoke(prompt)

    return response.content