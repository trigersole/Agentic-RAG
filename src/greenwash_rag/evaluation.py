from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openai import OpenAI

from .store import VectorStore

LABELS = ("none", "planning", "implemented", "indeterminate")

CLASSIFICATION_RULES = """TASK AND SCOPE
- Label what the focal sustainability statement expresses for the ONE proposed aspect. Ignore actions
  concerning other aspects in the same sentence.
- When no aspect is proposed, choose none if the sentence contains no valid sustainability aspect. If
  it does contain one, classify the action expressed for that aspect.
- Category is reference metadata, not the output label.

LABEL DEFINITIONS
- planning: the company has planned an action or made a future commitment to incorporate, address, or
  advance the aspect. This includes "strive to", "seek to", "endeavour to", aims and targets, and a
  forward-looking goal that depends on a forthcoming event.
- implemented: the company has already incorporated an activity or entity into operations, has already
  taken a concrete action, or has advanced or achieved a sustainability goal.
- indeterminate: the statement does not clearly establish whether or how the company engages with the
  aspect. This includes vague, non-committal, exploratory, non-attributable, or action-free language.
- none: there is no valid aspect-action pair. For a proposed aspect, use none only when that proposed
  span clearly is not a valid sustainability aspect; do not use none merely because its action is vague.

ASPECT VALIDITY
A valid aspect is an explicit, continuous phrase in the statement describing a sustainability entity,
goal, sub-area, or activity, directly linked to the supplied sustainability category without additional
assumptions. A slightly long or short span can still be valid.

APPLY THIS DECISION ORDER
1. Does the proposed aspect pass the aspect-validity rules? If not, label none.
2. Is the company, its board, management, staff, or subsidiary clearly the actor engaging with it? If
   not, label indeterminate.
3. Is the engagement hedged or exploratory—for example "where possible", "if feasible", "considering",
   or "exploring"? Label indeterminate. Hedging overrides past, present, or future tense.
4. Is the sentence only disclosing figures, listing responsibilities, expressing belief or risk, or
   describing an action being reduced or removed without a sustainability enhancement? Label
   indeterminate.
5. Has an action already been taken or a goal been advanced or achieved? Label implemented.
6. Has the company planned or committed to an action, including strive/seek/endeavour language or a
   future dependency? Label planning.
7. Otherwise label indeterminate. When uncertain, choose the single best-supported label.

IMPORTANT HARD CASES
- Monitoring or tracking that the company already performs is implemented; merely reporting the
  monitored numbers is indeterminate.
- "Zero incidents" or "no breaches" during a completed period counts as a goal achieved: implemented.
- Future disclosure of figures is still mere disclosure: indeterminate.
- A responsibility such as "the board is responsible for governance" without an improvement action is
  indeterminate.
- A framework that any company could follow is unattributed and therefore indeterminate.
- A completed or operating action described with "where possible" remains indeterminate.
- If "strives to" governs multiple coordinated actions, all governed actions are planning.

CONTEXT AND RETRIEVED EVIDENCE
Always label the focal statement itself. Neighboring context and retrieved passages may only resolve a
reference such as "this target" or "these measures", identify the actor, or clarify what the named
programme, policy, metric, or aspect means. They must not override the focal sentence. In particular, a
later passage saying a plan was completed must not change a planning focal statement to implemented.
Treat company reports as evidence of what the company disclosed, not independent proof of real-world
impact."""

CLASSIFIER_PROMPT = f"""Classify the sustainability-report statement into exactly one label using the
following annotation guide.

{CLASSIFICATION_RULES}

Return only one lowercase label: none, planning, implemented, or indeterminate.
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
    reasoning_effort: str,
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
        reasoning={"effort": reasoning_effort},
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

    metrics = {
        "items": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_by_label.values()) / len(LABELS) if total else 0.0,
        "f1_by_label": f1_by_label,
        "prediction_counts": dict(Counter(row["prediction"] for row in results)),
        "confusion_matrix": confusion,
    }
    target_exclusion = bool(results) and all(
        row.get("target_exclusion_enabled") for row in results
    )
    if target_exclusion:
        metrics.update(
            {
                "target_leakage_rate": retrieval_hits / total if total else 0.0,
                "retrieval_relevance_metrics": (
                    "not computed: the dataset identifies the focal-statement chunk, not judged "
                    "relevant evidence, and that chunk is deliberately excluded"
                ),
            }
        )
    else:
        metrics.update(
            {
                f"retrieval_recall@{top_k}": retrieval_hits / total if total else 0.0,
                "retrieval_mrr": reciprocal_rank / total if total else 0.0,
            }
        )
    return metrics


def run_evaluation(
    *,
    store: VectorStore,
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    embedding_model: str,
    items: list[dict[str, Any]],
    top_k: int,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queries = [evaluation_query(item) for item in items]
    vectors: list[list[float] | None]
    if store.retrieval_mode == "bm25":
        vectors = [None] * len(queries)
    else:
        vectors = embed_queries(client, embedding_model, queries)
    results: list[dict[str, Any]] = []

    for item, query, vector in zip(items, queries, vectors, strict=True):
        hits = store.search_vector(
            vector,
            query=query,
            exclude_text=str(item.get("text") or ""),
            top_k=top_k,
        )
        hit_dicts = [hit.model_dump() for hit in hits]
        relevant = gold_chunk_ids(store, item)
        ranks = [rank for rank, hit in enumerate(hits, start=1) if hit.chunk_id in relevant]
        prediction = classify(client, model, reasoning_effort, item, hit_dicts)
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
                "target_exclusion_enabled": True,
            }
        )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return compute_metrics(results, top_k), results
