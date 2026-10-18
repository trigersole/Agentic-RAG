# Greenwash Agentic RAG

An evidence-grounded, tool-using RAG service for the supplied Greenwash knowledge base. It uses the
existing `text-embedding-3-small` vectors for retrieval and lets an OpenAI model decide when to search
again, narrow by company/year, inspect corpus coverage, or compare disclosures.

## What is needed

1. Python 3.11+ and `uv`.
2. The knowledge-base directory containing `chunks.jsonl`, `embeddings.npy`, and `manifest.json`.
3. An OpenAI API key. Query embeddings must use `text-embedding-3-small`, matching the stored vectors.
4. Optional: the supplied evaluation JSONL under `data/eval/` for benchmark work.

The initial local NumPy backend is appropriate for 18,839 passages. PostgreSQL/pgvector becomes useful
for concurrent deployment, access control, or much larger corpora; it is not required for the first
working system.

## Setup

```powershell
uv sync --extra dev
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY and GREENWASH_DATA_DIR.
# OPENAI_CHAT_MODEL=gpt-5.6-luna and OPENAI_REASONING_EFFORT=low are the defaults.
uv run greenwash-rag doctor
uv run greenwash-rag validate
uv run greenwash-rag ask "Compare emissions targets across two banks"
```

Run one end-to-end evaluation smoke test, then a small dev-set baseline:

```powershell
uv run greenwash-rag smoke
uv run greenwash-rag evaluate --split dev --limit 25
```

Omit `--limit` to evaluate all 331 dev examples. Use the held-out test split only after tuning:
`uv run greenwash-rag evaluate --split test`. Evaluation reports classification accuracy, macro-F1,
retrieval Recall@k, retrieval MRR, and a confusion matrix; per-item outputs are saved under `runs/`.

To run the API:

```powershell
uv run uvicorn greenwash_rag.api:app --reload
```

Then POST JSON such as `{"question":"What evidence supports the company's net-zero claim?"}` to
`http://127.0.0.1:8000/ask`. Swagger documentation is at `/docs`.

## Data placement

Either copy the three files to `artifacts/baseline/dev/`, or leave them elsewhere and set an absolute
path in `GREENWASH_DATA_DIR`. The application never modifies the source corpus.

## Why this is agentic

A fixed RAG chain always performs one search. This system exposes read-only tools to the model. The
model can inspect companies and years, issue multiple differently phrased searches, add metadata
filters, and compare evidence before composing an answer. Tool steps are bounded, returned as a trace,
and all substantive claims must cite retrieved chunk IDs.

## Next production steps

- Run the frozen evaluation set and establish retrieval/answer baselines.
- Add a reranker only if evaluation shows top-k retrieval misses.
- Add authentication, request logging with redaction, rate limits, and persistent trace storage.
- Move vectors to pgvector if concurrency or corpus size makes the in-memory backend unsuitable.

