Litmus — RAG with Evaluation Harness

A versioned Retrieval-Augmented Generation (RAG) system with an evaluation harness for measuring retrieval quality, answer quality, and refusal behavior.

Litmus is developed incrementally from a baseline RAG pipeline to an adaptive RAG architecture. Each version is treated as a working engineering checkpoint and evaluated before moving to the next version.

Project Status

Version

Status

Capability

V1

Complete

Baseline RAG

V2

Complete

Evaluation Harness

V3

Planned

Multi-Query Retrieval

V4

Planned

Query Decomposition

V5

Planned

Multi-Source Router

V6

Planned

Corrective RAG

V7

Planned

Adaptive RAG + LangGraph

Overview

A basic RAG system follows:

Question → Retrieve → LLM → Answer

Litmus adds an evaluation layer:

Question
   ↓
Retrieve
   ↓
Evaluate Retrieval
   ↓
Generate Answer
   ↓
Evaluate Answer
   ↓
Analyze Failures
   ↓
Improve System

The goal is to measure whether retrieval and generation are actually working rather than relying only on subjective chatbot responses.

V1 — Baseline RAG

V1 implements the foundational RAG pipeline.

Architecture

PDF Documents
      ↓
PDF Loader
      ↓
Text Chunking
      ↓
Local Embeddings
      ↓
Chroma Vector Store
      ↓
Similarity Retriever
      ↓
Retrieved Context
      ↓
Groq LLM
      ↓
Answer + Sources

V1 Components

PDF Loader

File:

src/ingest/loader.py

Loads PDF documents and preserves the source filename in document metadata.

Document Chunking

File:

src/ingest/chunker.py

Uses:

RecursiveCharacterTextSplitter

Configuration:

chunk_size = 500
chunk_overlap = 50

Embeddings

File:

src/retrieval/embedder.py

Model:

sentence-transformers/all-MiniLM-L6-v2

Embeddings are generated locally.

Vector Store

File:

src/retrieval/vector_store.py

Vector database:

Chroma

The generated vector database is stored locally under:

data/processed/chroma/

This generated directory is excluded from Git.

Retriever

File:

src/retrieval/retriever.py

V1 uses similarity search with:

k = 4

LLM Client

File:

src/generation/llm_client.py

Current model:

openai/gpt-oss-20b

The model is accessed through the Groq OpenAI-compatible API.

The generation prompt is designed to:

Answer using retrieved context.

Avoid unsupported information.

Return grounded answers.

Refuse when sufficient information is unavailable.

RAG Pipeline

File:

src/pipeline.py

Main function:

answer_question(question)

Returns:

answer
sources

V1 Knowledge Base

The current knowledge base contains 8 PDF documents:

data/raw/
├── ats_optimization.pdf
├── cover_letters.pdf
├── interview_prep.pdf
├── job_search_strategy.pdf
├── linkedin_branding.pdf
├── remote_work_and_career_pivot.pdf
├── resume_writing.pdf
└── salary_negotiation.pdf

V2 — Evaluation Harness

V2 adds a formal evaluation layer on top of the V1 RAG pipeline.

The evaluation uses a 50-question golden dataset.

Dataset:

eval/golden_dataset.json

The dataset covers:

Resume Writing

ATS Optimization

Cover Letters

Interview Preparation

Job Search Strategy

LinkedIn Branding

Salary Negotiation

Remote Work and Career Pivot

The dataset contains answerable questions as well as an out-of-scope question for testing refusal behavior.

V2 Evaluation Architecture

                 Golden Dataset
                  50 Questions
                       │
                       ▼
                  V1 RAG
                  Pipeline
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Retrieval Metrics   Generation Metrics
              │                 │
              ▼                 ▼
       ┌──────────────┐   ┌──────────────────┐
       │ Hit Rate@4   │   │ Faithfulness     │
       │ MRR          │   │ Answer Relevancy │
       │ Context      │   │ Answer Correct.  │
       │ Precision    │   │ Refusal Accuracy │
       └──────────────┘   └──────────────────┘

V2 Retrieval Metrics

File:

eval/metrics.py

Hit Rate@4

Measures whether the expected source document appears in the top 4 retrieved documents.

Current baseline:

Hit Rate@4 = 0.98

Mean Reciprocal Rank (MRR)

Measures how highly the expected source document appears in the retrieval results.

Current baseline:

MRR = 0.95

Context Precision

Measures the proportion of retrieved documents that belong to the expected source set.

Current baseline:

Context Precision = 0.56

Results:

eval/results/v1_results.json

V2 Generation Metrics

Generation evaluation uses Ragas.

Faithfulness

Measures whether the generated answer is supported by the retrieved context.

Answer Relevancy

Measures whether the generated answer is relevant to the question.

Answer Correctness

Measures how closely the generated answer matches the expected ground truth.

Refusal Accuracy

A custom metric measuring whether the system correctly refuses questions when the required information is unavailable.

Results:

eval/results/v2_generation_results.json

V2 Evaluation Reliability

The Ragas evaluator is designed to avoid fabricating evaluation results.

Individual metrics can fail because of API or structured-output validation issues.

The evaluator therefore:

Evaluates metrics independently.

Retries failed metrics.

Preserves successful metric values.

Marks partially evaluated questions as partial.

Records the error.

Leaves unavailable metric values as null.

This keeps evaluation results transparent.

V2 Findings

Q037 — Retrieval Failure

Question:

What is the recommended first step before starting a job search?

The correct information exists in:

job_search_strategy.pdf

The correct chunk also exists in the vector store.

However, the original baseline query did not retrieve the correct chunk within the top results.

A more targeted query formulation was able to retrieve the correct chunk.

This indicates:

Correct document exists
        ↓
Correct chunk exists
        ↓
Baseline query misses it
        ↓
Query / retrieval formulation problem

This finding provides a concrete reason to introduce multi-query retrieval in V3.

Test Suite

Tests are located under:

tests/

Current test files:

tests/
├── test_chunker.py
├── test_embedder.py
├── test_golden_retrieval.py
├── test_ingest_all.py
├── test_llm_client.py
├── test_loader.py
├── test_metrics.py
├── test_pipeline.py
├── test_retriever.py
├── test_run_eval.py
└── test_vector_store.py

Test Coverage

Test

Purpose

test_chunker.py

Tests document chunking

test_embedder.py

Tests embedding generation

test_golden_retrieval.py

Tests retrieval against known golden questions

test_ingest_all.py

Tests PDF ingestion

test_llm_client.py

Tests LLM client

test_loader.py

Tests PDF loading

test_metrics.py

Tests evaluation metrics

test_pipeline.py

Tests end-to-end RAG pipeline

test_retriever.py

Tests retrieval

test_run_eval.py

Tests evaluation runner

test_vector_store.py

Tests Chroma vector store

Running Tests

Run all tests

python -m pytest -v

Run metric tests

python -m pytest tests/test_metrics.py -v

Run golden retrieval tests

python -m pytest tests/test_golden_retrieval.py -v

Run the main V2 verification tests

python -m pytest tests/test_metrics.py tests/test_golden_retrieval.py -v

Verified result:

16 passed

Installation

Requirements

Python 3.10+

Groq API key

Internet connection for LLM API access

Clone Repository

git clone https://github.com/vaibhavcodes-bit/litmus-rag-eval-harness.git
cd litmus-rag-eval-harness

Create Virtual Environment

Windows PowerShell:

python -m venv .venv

Activate:

.\.venv\Scripts\Activate.ps1

Install Dependencies

pip install -r requirements.txt

Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

Never commit the .env file.

The repository ignores:

.env
.venv/
__pycache__/
.pytest_cache/
data/processed/chroma/

Build the Vector Store

Run:

python src/ingest/ingest_all.py

This performs:

Load PDFs
    ↓
Split Documents
    ↓
Generate Embeddings
    ↓
Store Embeddings in Chroma

The generated database is stored under:

data/processed/chroma/

Run V1 Evaluation

Run:

python eval/run_eval.py --version v1

Results:

eval/results/v1_results.json

Run V2 Generation Evaluation

Run:

python eval/run_ragas_eval.py

The evaluator supports resumable execution.

For a limited test:

python eval/run_ragas_eval.py --retry-failed --limit 6

Results:

eval/results/v2_generation_results.json

Project Structure

litmus-rag-eval-harness/
│
├── README.md
├── requirements.txt
├── pytest.ini
│
├── data/
│   ├── raw/
│   │   ├── ats_optimization.pdf
│   │   ├── cover_letters.pdf
│   │   ├── interview_prep.pdf
│   │   ├── job_search_strategy.pdf
│   │   ├── linkedin_branding.pdf
│   │   ├── remote_work_and_career_pivot.pdf
│   │   ├── resume_writing.pdf
│   │   └── salary_negotiation.pdf
│   │
│   └── processed/
│       └── chroma/
│
├── src/
│   ├── generation/
│   │   └── llm_client.py
│   │
│   ├── ingest/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   └── ingest_all.py
│   │
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── pipeline.py
│   └── main.py
│
├── eval/
│   ├── golden_dataset.json
│   ├── metrics.py
│   ├── run_eval.py
│   ├── run_generation_eval.py
│   ├── run_ragas_eval.py
│   └── results/
│
└── tests/
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_golden_retrieval.py
    ├── test_ingest_all.py
    ├── test_llm_client.py
    ├── test_loader.py
    ├── test_metrics.py
    ├── test_pipeline.py
    ├── test_retriever.py
    ├── test_run_eval.py
    └── test_vector_store.py

Technology Stack

Component

Technology

Language

Python

LLM Provider

Groq

LLM Model

openai/gpt-oss-20b

PDF Loader

PyPDFLoader

Text Splitter

RecursiveCharacterTextSplitter

Embeddings

sentence-transformers/all-MiniLM-L6-v2

Vector Database

Chroma

RAG Components

LangChain

Evaluation

Ragas + Custom Metrics

Testing

Pytest

Version Control

Git / GitHub

Version Roadmap

V3 — Multi-Query Retrieval

Generate multiple query variations to improve retrieval recall.

Original Question
        ↓
Generate 3–4 Query Variations
        ↓
Parallel Retrieval
        ↓
Merge Results
        ↓
Deduplicate
        ↓
Generate Answer

Planned file:

src/retrieval/multi_query.py

Primary goals:

Improve recall.

Handle query wording differences.

Address retrieval failures such as Q037.

V4 — Query Decomposition

Designed for complex questions containing multiple information requirements.

Complex Question
       ↓
Sub-question 1
Sub-question 2
Sub-question 3
       ↓
Retrieve Evidence
       ↓
Combine Evidence
       ↓
Generate Answer

Planned file:

src/retrieval/decomposition.py

V5 — Multi-Source Router

The system will support multiple information sources:

             Question
                ↓
              Router
            /   |   \
           /    |    \
      Vector   SQL   Web

Planned files:

src/retrieval/router.py
src/retrieval/sql_source.py
src/retrieval/web_source.py

V6 — Corrective RAG

V6 introduces document grading and query correction.

Question
   ↓
Retrieve
   ↓
Grade Documents
   ↓
Relevant?
  /    \
Yes     No
 |       |
 ↓       ↓
Generate Rewrite Query
          ↓
       Retrieve

Maximum correction retries:

2

Planned file:

src/retrieval/grader.py

V7 — Adaptive RAG + LangGraph

V7 introduces an adaptive graph-based workflow.

Route
  ↓
Retrieve
  ↓
Grade Documents
  ↓
Generate
  ↓
Grade Answer
  ↓
Good?
 /   \
Yes   No
 |     |
 ↓     ↓
Final  Retry

Planned file:

src/graph/adaptive_rag_graph.py

LangGraph will be introduced at V7 for the adaptive workflow.

Engineering Approach

Litmus follows an evaluation-driven development process:

Build
  ↓
Test
  ↓
Evaluate
  ↓
Inspect Failures
  ↓
Identify Root Cause
  ↓
Implement Improvement
  ↓
Evaluate Again

The objective is to make every major RAG improvement measurable.

Failure Diagnosis

Retrieval Failure

The correct information exists but is not retrieved.

Possible solutions:

Better chunking

Multi-query retrieval

Query rewriting

Metadata filtering

Reranking

Generation / Grounding Failure

Relevant context is retrieved, but the generated answer is incorrect or unsupported.

Possible solutions:

Better prompts

Answer grading

Citations

Structured output

Precision Failure

Too many irrelevant chunks are retrieved.

Possible solutions:

Smaller K

Reranking

Metadata filtering

Single-Shot Retrieval Failure

One retrieval query is insufficient.

Possible solutions:

Multi-query retrieval

Query decomposition

Iterative retrieval

Corrective RAG

Current V2 Checkpoint

V1

PDF ingestion

Recursive text splitting

Local embeddings

Chroma vector store

Similarity retrieval

Groq LLM integration

Grounded answer generation

Source metadata

V2

50-question golden dataset

Retrieval evaluation

Hit Rate@4

MRR

Context Precision

Ragas generation evaluation

Faithfulness

Answer Relevancy

Answer Correctness

Refusal Accuracy

Resumable evaluation

Unit tests

Golden retrieval tests

16 verified V2 tests passing

GitHub repository

Baseline Results

Current V2 retrieval baseline:

Metric

Score

Hit Rate@4

0.98

MRR

0.95

Context Precision

0.56

These values establish the baseline for future versions.

V3 and later versions should be evaluated against this baseline to determine whether the additional retrieval strategies provide measurable improvements.

GitHub

Repository:

https://github.com/vaibhavcodes-bit/litmus-rag-eval-harness

Branch:

main

Current V2 commit:

5ff3732

Commit message:

Complete V2 evaluation harness

License

This project is currently maintained as a technical portfolio and engineering project.