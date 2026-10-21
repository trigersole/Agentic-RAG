# Greenwash Agentic RAG

An evidence-grounded, tool-using RAG service for the supplied Greenwash knowledge base. Every search
combines BM25 and the existing `text-embedding-3-small` vectors with reciprocal-rank fusion (RRF), then
reranks the leading candidates with a CPU cross-encoder. An OpenAI model decides when to search again,
rewrite the query, narrow by company/year, inspect corpus coverage, or compare disclosures.

## What is needed

1. Python 3.11 or 3.12 and `uv`.
2. The knowledge-base directory containing `chunks.jsonl`, `embeddings.npy`, and `manifest.json`.
3. An OpenAI API key. Query embeddings must use `text-embedding-3-small`, matching the stored vectors.
4. About 90 MB of disk space for the cross-encoder downloaded by FastEmbed on its first search.
5. Optional: the supplied evaluation JSONL under `data/eval/` for benchmark work.

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

The default retrieval pipeline is:

```text
metadata filters -> BM25 top 50 + dense top 50 -> RRF (k=60)
                 -> cross-encoder rerank top 20 -> deduplicate -> top k
```

Set `GREENWASH_RETRIEVAL_MODE` to `bm25`, `dense`, `rrf`, or `rrf+rerank` for an ablation. The default
reranker is `Xenova/ms-marco-MiniLM-L-6-v2` through FastEmbed/ONNX, so PyTorch and a GPU are not needed.

Run one end-to-end evaluation smoke test, then a small dev-set baseline:

```powershell
uv run greenwash-rag smoke
uv run greenwash-rag evaluate --split dev --limit 25
```

Compare the fixed baseline with the full tool-using classification agent:

```powershell
# One-item agentic smoke test
uv run greenwash-rag evaluate --mode agentic --split dev --limit 1 --output runs/agentic_smoke.jsonl

# Small repeated evaluation (odd repeat counts avoid majority-vote ties)
uv run greenwash-rag evaluate --mode agentic --split dev --limit 25 --repeats 3 --output runs/agentic_dev25.jsonl
```

Agentic mode lets the model choose corpus tools and repeat hybrid searches before returning a strict
JSON classification. Each search is ranked deterministically by BM25/dense/RRF/reranking; the agent
plans and rewrites queries but does not calculate retrieval scores. Evaluation searches automatically
remove passages containing the focal statement to prevent target leakage. Agentic evaluation also
reports majority-vote accuracy, consistency, tool/search usage, evidence validity, latency, token
usage, and errors. It costs more than fixed mode because each item may require several API calls.

Omit `--limit` to evaluate all 331 dev examples. Use the held-out test split only after tuning:
`uv run greenwash-rag evaluate --split test`. Evaluation reports classification accuracy, macro-F1,
target-leakage rate, and a confusion matrix; per-item outputs are saved under `runs/`. Retrieval Recall
and MRR require separately judged relevance labels—the dataset's target-containing chunk is excluded,
so it is not treated as relevant evidence.

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

## Retrieval configuration

| Variable | Default | Meaning |
|---|---:|---|
| `GREENWASH_RETRIEVAL_MODE` | `rrf+rerank` | retrieval ablation mode |
| `GREENWASH_RETRIEVAL_CANDIDATES` | `50` | candidates from BM25 and dense search |
| `GREENWASH_RERANK_DEPTH` | `20` | fused candidates scored by the cross-encoder |
| `GREENWASH_RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | FastEmbed reranker |

The search tool also supports exact year, year range, excluded year, and company filters. Normalized
duplicate passages are removed before results are returned.

## Next production steps

- Run the frozen evaluation set and establish retrieval/answer baselines.
- Collect human relevance judgements so retrieval quality can be measured independently of label accuracy.
- Add authentication, request logging with redaction, rate limits, and persistent trace storage.
- Move vectors to pgvector if concurrency or corpus size makes the in-memory backend unsuitable.

