import json

import numpy as np
import pytest

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
