from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openai import OpenAI

from .store import VectorStore

LABELS = ("none", "planning", "implemented", "indeterminate")

CLASSIFIER_PROMPT = """Classify the sustainability-report statement into exactly one label.

Labels:
- none: no environmental/social action or commitment concerning the named aspect is expressed.
- planning: a future intention, aim, target, commitment, or proposed action is expressed, without clear
  evidence in the supplied text that the action has already been carried out.
- implemented: a concrete action, policy, process, measurement, disclosure, or initiative has already
  been carried out or is currently operating.
- indeterminate: the wording is contextual, aspirational, vague, or otherwise insufficient to decide
  whether a relevant action is planned or implemented.

Use the focal statement, its neighboring context, and retrieved report evidence. Reported corporate
claims are evidence of what was disclosed, not proof that real-world impact occurred. Return only one
lowercase label: none, planning, implemented, or indeterminate.
"""


def load_items(path: Path, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
    if split != "all":
        items = [item for item in items if item.get("split") == split]
    return items[:limit] if limit is not None else items


def evaluation_query(item: dict[str, Any]) -> str:
    return (
        f"Statement: {item['text']}\n"
        f"Aspect: {item.get('aspect') or 'not specified'}\n"
        f"Category: {item.get('category') or 'not specified'}"
    )


def gold_chunk_ids(store: VectorStore, item: dict[str, Any]) -> set[str]:
    """Derive relevant chunks from exact statement containment and report metadata."""
    company = str(item.get("company", "")).casefold()
    year = str(item.get("year", ""))
    page = str(item.get("page", ""))
    statement = str(item["text"])
    candidates = [
        chunk
        for chunk in store.chunks
        if str(chunk.get("company", "")).casefold() == company
        and str(chunk.get("year", "")) == year
        and str(chunk.get("page", "")) == page
    ]
    matches = {str(chunk["chunk_id"]) for chunk in candidates if statement in chunk["text"]}
    if not matches:
        matches = {
            str(chunk["chunk_id"])
            for chunk in store.chunks
            if statement in str(chunk.get("text", ""))
        }
    return matches


def classify(
    client: OpenAI,
    model: str,
    item: dict[str, Any],
    evidence: Iterable[dict[str, Any]],
) -> str:
    evidence_text = "\n\n".join(
        f"[{hit['chunk_id']} | {hit['company']} | {hit['year']} | p. {hit['page']}]\n{hit['text']}"
        for hit in evidence
    )
    user_input = (
        f"Focal statement:\n{item['text']}\n\n"
        f"Aspect: {item.get('aspect') or 'not specified'}\n"
        f"Category: {item.get('category') or 'not specified'}\n\n"
        f"Previous context:\n{item.get('context_prev') or ''}\n\n"
        f"Next context:\n{item.get('context_next') or ''}\n\n"
        f"Retrieved evidence:\n{evidence_text}"
    )
    response = client.responses.create(
        model=model,
        instructions=CLASSIFIER_PROMPT,
        input=user_input,
    )
    prediction = response.output_text.strip().lower().strip("` .\n\t")
    if prediction not in LABELS:
        for label in LABELS:
            if label in prediction:
                return label
        return "invalid"
    return prediction


def embed_queries(
    client: OpenAI, model: str, queries: list[str], batch_size: int = 64
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(queries), batch_size):
        response = client.embeddings.create(
            model=model,
            input=queries[start : start + batch_size],
            encoding_format="float",
        )
        vectors.extend(
            entry.embedding for entry in sorted(response.data, key=lambda entry: entry.index)
        )
    return vectors


def compute_metrics(results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    total = len(results)
    correct = sum(row["prediction"] == row["gold_label"] for row in results)
    retrieval_hits = sum(bool(row["retrieved_gold_ranks"]) for row in results)
    reciprocal_rank = sum(
        1.0 / min(row["retrieved_gold_ranks"]) if row["retrieved_gold_ranks"] else 0.0
        for row in results
    )
    confusion = {gold: {pred: 0 for pred in (*LABELS, "invalid")} for gold in LABELS}
    for row in results:
        confusion[row["gold_label"]][row["prediction"]] += 1

    f1_by_label: dict[str, float] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in confusion[label] if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_by_label[label] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    return {
        "items": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_by_label.values()) / len(LABELS) if total else 0.0,
        "f1_by_label": f1_by_label,
        f"retrieval_recall@{top_k}": retrieval_hits / total if total else 0.0,
        "retrieval_mrr": reciprocal_rank / total if total else 0.0,
        "prediction_counts": dict(Counter(row["prediction"] for row in results)),
        "confusion_matrix": confusion,
    }


def run_evaluation(
    *,
    store: VectorStore,
    client: OpenAI,
    model: str,
    embedding_model: str,
    items: list[dict[str, Any]],
    top_k: int,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queries = [evaluation_query(item) for item in items]
    vectors = embed_queries(client, embedding_model, queries)
    results: list[dict[str, Any]] = []

    for item, vector in zip(items, vectors, strict=True):
        hits = store.search_vector(vector, top_k=top_k)
        hit_dicts = [hit.model_dump() for hit in hits]
        relevant = gold_chunk_ids(store, item)
        ranks = [rank for rank, hit in enumerate(hits, start=1) if hit.chunk_id in relevant]
        prediction = classify(client, model, item, hit_dicts)
        results.append(
            {
                "item_id": item["item_id"],
                "split": item.get("split"),
                "gold_label": item["gold_label"],
                "prediction": prediction,
                "correct": prediction == item["gold_label"],
                "gold_chunk_ids": sorted(relevant),
                "retrieved_chunk_ids": [hit.chunk_id for hit in hits],
                "retrieved_gold_ranks": ranks,
            }
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return compute_metrics(results, top_k), results
