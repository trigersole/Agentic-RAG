from greenwash_rag.agentic_evaluation import AGENTIC_CLASSIFIER_PROMPT, compute_agentic_metrics
from greenwash_rag.evaluation import CLASSIFICATION_RULES, CLASSIFIER_PROMPT, compute_metrics


def test_fixed_and_agentic_prompts_share_complete_classification_rules():
    assert CLASSIFICATION_RULES in CLASSIFIER_PROMPT
    assert CLASSIFICATION_RULES in AGENTIC_CLASSIFIER_PROMPT
    for required_rule in (
        "Ignore actions",
        "concerning other aspects",
        "Hedging overrides",
        '"seek to"',
        "Monitoring or tracking",
        "Zero incidents",
        "reduced or removed",
        "later passage saying a plan was completed",
    ):
        assert required_rule in CLASSIFICATION_RULES


def test_compute_metrics():
    results = [
        {
            "gold_label": "planning",
            "prediction": "planning",
            "retrieved_gold_ranks": [1],
        },
        {
            "gold_label": "implemented",
            "prediction": "planning",
            "retrieved_gold_ranks": [],
        },
    ]
    metrics = compute_metrics(results, top_k=8)
    assert metrics["accuracy"] == 0.5
    assert metrics["retrieval_recall@8"] == 0.5
    assert metrics["retrieval_mrr"] == 0.5
    assert metrics["confusion_matrix"]["implemented"]["planning"] == 1


def test_agentic_metrics_report_tools_evidence_and_vote_ties():
    base = {
        "item_id": "1",
        "gold_label": "planning",
        "correct": True,
        "retrieved_gold_ranks": [1],
        "trace": [{"tool": "search_reports"}],
        "evidence_chunk_ids": ["chunk-1"],
        "retrieved_chunk_ids": ["chunk-1"],
        "latency_seconds": 1.0,
        "input_tokens": 10,
        "output_tokens": 2,
        "error": None,
    }
    results = [
        {**base, "repeat": 1, "prediction": "planning"},
        {**base, "repeat": 2, "prediction": "implemented", "correct": False},
    ]
    metrics = compute_agentic_metrics(results, top_k=8, repeats=2)
    assert metrics["accuracy"] == 0.5
    assert metrics["search_use_rate"] == 1.0
    assert metrics["evidence_validity_rate"] == 1.0
    assert metrics["majority_vote_tie_rate"] == 1.0
    assert metrics["majority_vote_accuracy"] == 0.0
