import json

import numpy as np
import pytest

from greenwash_rag.retrieval import CrossEncoderReranker, reciprocal_rank_fusion
from greenwash_rag.store import CorpusError, VectorStore


def make_corpus(tmp_path):
    chunks = [
        {
            "chunk_id": "a",
            "text": "solar renewable target",
            "company": "Alpha",
            "year": 2023,
            "page": 4,
        },
        {
            "chunk_id": "b",
            "text": "water use reduction",
            "company": "Beta",
            "year": 2022,
            "page": 7,
        },
        {
            "chunk_id": "c",
            "text": "emissions baseline",
            "company": "Alpha",
            "year": 2022,
            "page": 9,
        },
    ]
    (tmp_path / "chunks.jsonl").write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks), encoding="utf-8"
    )
    np.save(tmp_path / "embeddings.npy", np.eye(3, dtype=np.float32))
    (tmp_path / "manifest.json").write_text(json.dumps({"build_id": "test"}), encoding="utf-8")


def test_search_and_filters(tmp_path):
    make_corpus(tmp_path)
    store = VectorStore(tmp_path, lambda _: [1.0, 0.0, 0.0])
    assert store.search("solar", top_k=1)[0].chunk_id == "a"
    assert store.search("anything", company="Beta", top_k=3)[0].chunk_id == "b"
    assert store.search("anything", company="Alpha", year=2099) == []


def test_wrong_query_dimension(tmp_path):
    make_corpus(tmp_path)
    store = VectorStore(tmp_path, lambda _: [1.0, 0.0])
    with pytest.raises(CorpusError, match="shape"):
        store.search("bad dimensions")


def test_missing_files(tmp_path):
    with pytest.raises(CorpusError, match="Missing"):
        VectorStore(tmp_path, lambda _: [])


def test_bm25_mode_does_not_call_embedding_api(tmp_path):
    make_corpus(tmp_path)

    def should_not_embed(_query):
        raise AssertionError("BM25 must not request a query embedding")

    store = VectorStore(tmp_path, should_not_embed, retrieval_mode="bm25")
    hit = store.search("water reduction", top_k=1)[0]
    assert hit.chunk_id == "b"
    assert hit.retrieval_method == "bm25"


def test_rrf_combines_rank_positions_not_raw_scores():
    fused = reciprocal_rank_fusion([(1, 1000.0), (2, 999.0)], [(2, 0.1)], k0=60)
    assert fused[0][0] == 2


def test_hybrid_reranker_can_promote_lexical_candidate(tmp_path):
    make_corpus(tmp_path)
    reranker = CrossEncoderReranker(
        score_documents=lambda _query, documents: [
            10.0 if "water" in document else 0.0 for document in documents
        ]
    )
    store = VectorStore(
        tmp_path,
        lambda _: [1.0, 0.0, 0.0],
        retrieval_mode="rrf+rerank",
        candidate_k=3,
        rerank_depth=3,
        reranker=reranker,
    )
    hit = store.search("water reduction", top_k=1)[0]
    assert hit.chunk_id == "b"
    assert hit.retrieval_method == "rrf+rerank"


def test_search_filters_year_ranges_and_target_text(tmp_path):
    make_corpus(tmp_path)
    store = VectorStore(tmp_path, lambda _: [1.0, 0.0, 0.0])
    hits = store.search(
        "solar",
        company="Alpha",
        year_from=2022,
        year_to=2023,
        exclude_year=2022,
        exclude_text="solar renewable target",
    )
    assert hits == []
