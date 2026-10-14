from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .models import SearchHit


class CorpusError(RuntimeError):
    """Raised when the supplied corpus is missing or internally inconsistent."""


class VectorStore:
    REQUIRED_FILES = ("chunks.jsonl", "embeddings.npy", "manifest.json")

    def __init__(self, data_dir: Path, embed_query: Callable[[str], list[float]]):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.embed_query = embed_query
        self.manifest: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.vectors: np.ndarray
        self._companies: dict[str, set[str]] = defaultdict(set)
        self._load()

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
        top_k: int = 8,
    ) -> list[SearchHit]:
        return self.search_vector(self.embed_query(query), company=company, year=year, top_k=top_k)

    def search_vector(
        self,
        query_vector: list[float] | np.ndarray,
        *,
        company: str | None = None,
        year: int | str | None = None,
        top_k: int = 8,
    ) -> list[SearchHit]:
        """Search with a precomputed vector, useful for batched evaluation."""
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

        indices = np.flatnonzero(mask)
        if indices.size == 0:
            return []
        scores = np.asarray(self.vectors[indices] @ vector)
        count = min(top_k, indices.size)
        candidate_positions = np.argpartition(scores, -count)[-count:]
        ranked_positions = candidate_positions[np.argsort(scores[candidate_positions])[::-1]]

        results: list[SearchHit] = []
        for position in ranked_positions:
            index = int(indices[position])
            chunk = self.chunks[index]
            results.append(
                SearchHit(
                    chunk_id=str(chunk["chunk_id"]),
                    text=str(chunk["text"]),
                    company=str(chunk.get("company", "Unknown")),
                    year=chunk.get("year"),
                    page=chunk.get("page"),
                    file=chunk.get("file"),
                    statement_ids=chunk.get("statement_ids") or [],
                    score=round(float(scores[position]), 6),
                )
            )
        return results
