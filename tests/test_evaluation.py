from greenwash_rag.evaluation import compute_metrics


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
