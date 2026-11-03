"""Hybrid retrieval: weighted fusion of TF-IDF and dense (embedding) scores."""
import numpy as np


class HybridRetriever:
    name = "hybrid"

    def __init__(self, tfidf, dense, alpha=0.5):
        """alpha = weight of the dense score (1 - alpha goes to TF-IDF)."""
        self.tfidf, self.dense, self.alpha = tfidf, dense, alpha

    def scores(self, query: str) -> np.ndarray:
        return self.alpha * self.dense.scores(query) + (1 - self.alpha) * self.tfidf.scores(query)

    def search(self, query: str, top_k: int = 3):
        s = self.scores(query)
        top = np.argsort(-s)[:top_k]
        return [(int(i), float(s[i])) for i in top]
