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
    return {
        "status": "ok",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    print("DEBUG: /ask endpoint received request", flush=True)

    print(
        f"DEBUG: question={request.question!r}, "
        f"k={request.k}, "
        f"mode={request.mode!r}",
        flush=True,
    )

    result = answer_question(
        question=request.question,
        k=request.k,
        mode=request.mode,
    )

    print("DEBUG: answer_question completed", flush=True)

    print(
        f"DEBUG: sources returned={len(result.get('sources', []))}",
        flush=True,
    )

    print("DEBUG: /ask endpoint completed", flush=True)

    return result


@app.get("/eval/latest", response_model=EvalResponse)
def get_latest_evaluation():
    result_files = list(EVAL_RESULTS_DIR.glob("*.json"))

    if not result_files:
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found.",
        )

    latest_file = max(
        result_files,
        key=lambda file: file.stat().st_mtime,
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
            detail=f"Invalid evaluation JSON: {latest_file.name}",
        ) from exc

    return result