#!/bin/sh

set -e

echo "Checking Chroma database..."

if [ ! -f "/app/data/processed/chroma/chroma.sqlite3" ]; then
    echo "ERROR: Chroma database not found."
    echo "The Docker image must contain the pre-built Chroma database."
    exit 1
fi

echo "Chroma database found."
echo "Starting FastAPI..."

exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT:-10000}"