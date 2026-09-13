#!/bin/sh

set -e

echo "Checking Chroma database..."

if [ ! -f "/app/data/processed/chroma/chroma.sqlite3" ]; then
    echo "Chroma database not found."
    echo "Running PDF ingestion..."
    python -m src.ingest.ingest_all
else
    echo "Chroma database already exists."
fi

echo "Starting FastAPI..."

exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-10000}"