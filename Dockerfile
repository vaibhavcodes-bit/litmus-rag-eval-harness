FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src ./src
COPY data/raw ./data/raw
COPY data/processed/chroma ./data/processed/chroma
COPY eval ./eval
COPY start.sh ./start.sh

RUN chmod +x ./start.sh

EXPOSE 10000

CMD ["./start.sh"]
