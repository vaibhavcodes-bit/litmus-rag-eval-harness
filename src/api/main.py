from pathlib import Path
import json

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    AskRequest,
    AskResponse,
    EvalResponse,
)
from src.pipeline import answer_question


app = FastAPI(
    title="Litmus RAG API",
    description="FastAPI backend for the Litmus RAG evaluation system.",
    version="1.0.0",
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


@app.get("/health")
def health():
    print("DEBUG: /health endpoint received request", flush=True)

    return {
        "status": "ok",
    }


@app.post("/debug/post")
def debug_post():
    """
    Simple POST connectivity test.

    This endpoint does not call Chroma,
    embeddings, LangChain, Groq, or the RAG pipeline.
    """

    print("DEBUG: /debug/post received", flush=True)

    return {
        "status": "ok",
        "message": "POST request reached FastAPI",
    }


@app.post("/debug/body")
def debug_body(payload: dict):
    """
    Test JSON request-body parsing without touching
    AskRequest validation or the RAG pipeline.
    """

    print("DEBUG: /debug/body received", flush=True)

    print(
        f"DEBUG: payload={payload!r}",
        flush=True,
    )

    return {
        "status": "ok",
        "message": "JSON body reached FastAPI",
        "payload": payload,
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Main RAG question-answering endpoint.
    """

    print("DEBUG: /ask endpoint received request", flush=True)

    print(
        f"DEBUG: question={request.question!r}, "
        f"k={request.k}, "
        f"mode={request.mode!r}",
        flush=True,
    )

    print("DEBUG: calling answer_question()", flush=True)

    result = answer_question(
        question=request.question,
        k=request.k,
        mode=request.mode,
    )

    print("DEBUG: answer_question completed", flush=True)

    print(
        f"DEBUG: sources returned="
        f"{len(result.get('sources', []))}",
        flush=True,
    )

    print("DEBUG: /ask endpoint completed", flush=True)

    return result


@app.get("/eval/latest", response_model=EvalResponse)
def get_latest_evaluation():
    """
    Return the most recently generated evaluation result.
    """

    print(
        "DEBUG: /eval/latest endpoint received request",
        flush=True,
    )

    result_files = list(
        EVAL_RESULTS_DIR.glob("*.json")
    )

    if not result_files:
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found.",
        )

    latest_file = max(
        result_files,
        key=lambda file: file.stat().st_mtime,
    )

    print(
        f"DEBUG: latest evaluation file="
        f"{latest_file.name}",
        flush=True,
    )

    try:
        with latest_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            result = json.load(file)

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Invalid evaluation JSON: "
                f"{latest_file.name}"
            ),
        ) from exc

    print("DEBUG: /eval/latest completed", flush=True)

    return result
@app.post("/debug/ask-entry")
def debug_ask_entry(request: AskRequest):
    """
    Test AskRequest validation and FastAPI request handling
    without calling the RAG pipeline.
    """

    print("DEBUG: /debug/ask-entry received", flush=True)

    print(
        f"DEBUG: question={request.question!r}, "
        f"k={request.k}, "
        f"mode={request.mode!r}",
        flush=True,
    )

    return {
        "status": "ok",
        "message": "AskRequest reached FastAPI",
        "question": request.question,
        "k": request.k,
        "mode": request.mode,
    }
