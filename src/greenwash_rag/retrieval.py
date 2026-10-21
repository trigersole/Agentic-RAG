from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Small deterministic tokenizer shared by indexing and querying."""
    return TOKEN_RE.findall(text.casefold())


class BM25Index:
    """Okapi BM25 over an in-memory inverted index."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [tokens(document) for document in documents]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, document in enumerate(self.documents):
            for term, frequency in Counter(document).items():
                self.postings[term].append((index, frequency))
        self.document_count = len(self.documents)

    def search(
        self, query: str, *, allowed: set[int] | None = None, top_k: int = 50
    ) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokens(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            document_frequency = len(posting)
            inverse_frequency = math.log(
                1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for index, frequency in posting:
                if allowed is not None and index not in allowed:
                    continue
                length_normalisation = 1 - self.b + self.b * self.lengths[index] / (
                    self.average_length or 1.0
                )
                scores[index] += (
                    inverse_frequency
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * length_normalisation)
                )
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def reciprocal_rank_fusion(
    *rankings: Sequence[tuple[int, float]], k0: int = 60
) -> list[tuple[int, float]]:
    """Fuse rankings by rank so BM25 and cosine score scales never need calibration."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (document_id, _score) in enumerate(ranking, start=1):
            fused[document_id] += 1.0 / (k0 + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class CrossEncoderReranker:
    """Lazy ONNX cross-encoder wrapper; the model is downloaded only on first reranked search."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        score_documents: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
    ):
        self.model_name = model_name
        self._score_documents = score_documents
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        if self._score_documents is not None:
            return [float(score) for score in self._score_documents(query, documents)]
        return [float(score) for score in self._load().rerank(query, list(documents))]
