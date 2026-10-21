from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .models import SearchHit
from .retrieval import BM25Index, CrossEncoderReranker, reciprocal_rank_fusion

RetrievalMode = Literal["bm25", "dense", "rrf", "rrf+rerank"]
RETRIEVAL_MODES = ("bm25", "dense", "rrf", "rrf+rerank")
_SPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _SPACE_RE.sub(" ", text.casefold()).strip()


class CorpusError(RuntimeError):
    """Raised when the supplied corpus is missing or internally inconsistent."""


class VectorStore:
    REQUIRED_FILES = ("chunks.jsonl", "embeddings.npy", "manifest.json")

    def __init__(
        self,
        data_dir: Path,
        embed_query: Callable[[str], list[float]],
        *,
        retrieval_mode: RetrievalMode = "dense",
        candidate_k: int = 50,
        rerank_depth: int = 20,
        rrf_k: int = 60,
        reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        reranker: CrossEncoderReranker | None = None,
    ):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.embed_query = embed_query
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}")
        self.retrieval_mode = retrieval_mode
        self.candidate_k = max(1, candidate_k)
        self.rerank_depth = max(1, rerank_depth)
        self.rrf_k = max(1, rrf_k)
        self.reranker = reranker
        if retrieval_mode == "rrf+rerank" and self.reranker is None:
            self.reranker = CrossEncoderReranker(reranker_model)
        self.manifest: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.vectors: np.ndarray
        self._companies: dict[str, set[str]] = defaultdict(set)
        self._load()
        self.bm25 = BM25Index([self._indexed_text(chunk) for chunk in self.chunks])

    @staticmethod
    def _indexed_text(chunk: dict[str, Any]) -> str:
        """Add provenance to lexical search, matching the contextual text used by the corpus."""
        return (
            f"{chunk.get('company', 'Unknown')} ({chunk.get('year', '')}), "
            f"page {chunk.get('page', '')}: {chunk.get('text', '')}"
        )

    def _load(self) -> None:
        missing = [name for name in self.REQUIRED_FILES if not (self.data_dir / name).is_file()]
        if missing:
            raise CorpusError(
                f"Missing {', '.join(missing)} in {self.data_dir}. "
                "Set GREENWASH_DATA_DIR to the extracted knowledge-base directory."
            )

        self.manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))
        with (self.data_dir / "chunks.jsonl").open(encoding="utf-8") as handle:
            self.chunks = [json.loads(line) for line in handle if line.strip()]
        self.vectors = np.load(self.data_dir / "embeddings.npy", mmap_mode="r")

        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.chunks):
            raise CorpusError(
                f"Corpus mismatch: {len(self.chunks)} chunks but embeddings shape is "
                f"{self.vectors.shape}."
            )
        for chunk in self.chunks:
            self._companies[str(chunk.get("company", "Unknown"))].add(str(chunk.get("year", "")))

    def stats(self) -> dict[str, Any]:
        text_hash = hashlib.sha256(
            "".join(str(chunk.get("text", "")) for chunk in self.chunks).encode("utf-8")
        ).hexdigest()[:12]
        return {
            "data_dir": str(self.data_dir),
            "chunks": len(self.chunks),
            "embedding_shape": list(self.vectors.shape),
            "companies": len(self._companies),
            "retrieval": {
                "mode": self.retrieval_mode,
                "candidate_k": self.candidate_k,
                "rrf_k": self.rrf_k,
                "rerank_depth": self.rerank_depth,
                "reranker_model": getattr(self.reranker, "model_name", None),
            },
            "computed_concat_text_hash": text_hash,
            "manifest": self.manifest,
        }

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {"company": company, "years": sorted(year for year in years if year)}
            for company, years in sorted(self._companies.items())
        ]

    def search(
        self,
        query: str,
        *,
        company: str | None = None,
        year: int | str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        exclude_year: int | str | None = None,
        exclude_text: str | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        vector = self.embed_query(query) if self.retrieval_mode != "bm25" else None
        return self.search_vector(
            vector,
            query=query,
            company=company,
            year=year,
            year_from=year_from,
            year_to=year_to,
            exclude_year=exclude_year,
            exclude_text=exclude_text,
            top_k=top_k,
        )

    def search_vector(
        self,
        query_vector: list[float] | np.ndarray | None,
        *,
        query: str | None = None,
        company: str | None = None,
        year: int | str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        exclude_year: int | str | None = None,
        exclude_text: str | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        """Run sparse, dense, fused, or fused-and-reranked retrieval."""
        mask = np.ones(len(self.chunks), dtype=bool)
        if company:
            needle = company.casefold()
            mask &= np.fromiter(
                (needle in str(chunk.get("company", "")).casefold() for chunk in self.chunks),
                dtype=bool,
                count=len(self.chunks),
            )
        if year is not None:
            target_year = str(year)
            mask &= np.fromiter(
                (str(chunk.get("year", "")) == target_year for chunk in self.chunks),
                dtype=bool,
                count=len(self.chunks),
            )
        if year_from is not None:
            mask &= np.fromiter(
                (int(chunk.get("year", 0) or 0) >= year_from for chunk in self.chunks),
                dtype=bool,
                count=len(self.chunks),
            )
        if year_to is not None:
            mask &= np.fromiter(
                (int(chunk.get("year", 0) or 0) <= year_to for chunk in self.chunks),
                dtype=bool,
                count=len(self.chunks),
            )
        if exclude_year is not None:
            excluded_year = str(exclude_year)
            mask &= np.fromiter(
                (str(chunk.get("year", "")) != excluded_year for chunk in self.chunks),
                dtype=bool,
                count=len(self.chunks),
            )

        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return []

        allowed = {int(index) for index in indices}
        depth = min(max(top_k, self.candidate_k, self.rerank_depth), indices.size)
        dense_ranked: list[tuple[int, float]] = []
        if self.retrieval_mode != "bm25":
            if query_vector is None:
                raise CorpusError("A query vector is required for dense or hybrid retrieval.")
            vector = np.asarray(query_vector, dtype=np.float32)
            if vector.shape != (self.vectors.shape[1],):
                raise CorpusError(
                    f"Query embedding has shape {vector.shape}; expected ({self.vectors.shape[1]},). "
                    "Use the same embedding model and dimensions as the corpus."
                )
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0:
                raise CorpusError("Query embedding is empty or invalid.")
            vector /= norm
            dense_scores = np.asarray(self.vectors[indices] @ vector)
            count = min(depth, indices.size)
            positions = np.argpartition(dense_scores, -count)[-count:]
            positions = positions[np.argsort(dense_scores[positions])[::-1]]
            dense_ranked = [(int(indices[pos]), float(dense_scores[pos])) for pos in positions]

        sparse_ranked: list[tuple[int, float]] = []
        if self.retrieval_mode != "dense":
            if not query:
                raise CorpusError("Query text is required for BM25 or hybrid retrieval.")
            sparse_ranked = self.bm25.search(query, allowed=allowed, top_k=depth)

        if self.retrieval_mode == "bm25":
            ranked = sparse_ranked
        elif self.retrieval_mode == "dense":
            ranked = dense_ranked
        else:
            ranked = reciprocal_rank_fusion(sparse_ranked, dense_ranked, k0=self.rrf_k)

        if self.retrieval_mode == "rrf+rerank" and ranked:
            assert self.reranker is not None
            head = ranked[: self.rerank_depth]
            reranker_scores = self.reranker.rerank(
                query or "", [str(self.chunks[index].get("text", "")) for index, _ in head]
            )
            if len(reranker_scores) != len(head):
                raise CorpusError(
                    f"Reranker returned {len(reranker_scores)} scores for {len(head)} passages."
                )
            ranked = sorted(
                zip((index for index, _ in head), reranker_scores, strict=True),
                key=lambda item: (-item[1], item[0]),
            ) + ranked[self.rerank_depth :]

        results: list[SearchHit] = []
        seen: set[str] = set()
        excluded = _normalise(exclude_text or "")
        for index, score in ranked:
            chunk = self.chunks[index]
            normalised = _normalise(str(chunk.get("text", "")))
            if normalised in seen:
                continue
            if excluded and excluded in normalised:
                continue
            seen.add(normalised)
            results.append(
                SearchHit(
                    chunk_id=str(chunk["chunk_id"]),
                    text=str(chunk["text"]),
                    company=str(chunk.get("company", "Unknown")),
                    year=chunk.get("year"),
                    page=chunk.get("page"),
                    file=chunk.get("file"),
                    statement_ids=chunk.get("statement_ids") or [],
                    score=round(float(score), 6),
                    retrieval_method=self.retrieval_mode,
                )
            )
            if len(results) >= top_k:
                break
        return results
