import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():
    """
    Create the Groq LLM client.

    Required environment variable:
    GROQ_API_KEY (format: gsk_xxxxx...)

    Raises:
        ValueError: If GROQ_API_KEY is not set or has invalid format
        RuntimeError: If the Groq client cannot be initialized
    """

    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please add it to the .env file or set it as an environment variable."
        )

    # Validate API key format
    if not api_key.startswith("gsk_"):
        raise ValueError(
            "Invalid GROQ_API_KEY format. Expected key to start with "
            "'gsk_', got: "
            f"{api_key[:10]}..."
        )

    try:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            api_key=api_key,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Groq client: {str(e)}"
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

    Args:
        question: The user's question
        context: The context to base the answer on

    Returns:
        str: The LLM's response

    Raises:
        ValueError: If question or context is empty
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