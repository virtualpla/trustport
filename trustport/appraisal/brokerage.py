from __future__ import annotations

from collections.abc import Sequence

from rank_bm25 import BM25Okapi


class Bm25Retriever:
    def __init__(self, corpus: Sequence[Sequence[int]]) -> None:
        self.corpus = [list(doc) for doc in corpus]
        tokenized = [[str(tok) for tok in doc] for doc in self.corpus]
        self.index = BM25Okapi(tokenized)

    def retrieve(self, query: Sequence[int], top_k: int) -> list[tuple[int, float]]:
        tokens = [str(tok) for tok in query]
        scores = self.index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]
