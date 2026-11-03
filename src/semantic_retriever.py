"""Semantic retrieval using Sentence Transformers embeddings."""
import numpy as np
from sentence_transformers import SentenceTransformer

from .preprocess import light_clean


class SemanticRetriever:
    name = "semantic"

    def __init__(self, questions, model_name="all-MiniLM-L6-v2"):
        self.questions = list(questions)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        cleaned = [light_clean(q) for q in self.questions]
        # normalized embeddings -> dot product == cosine similarity
        self.embeddings = self.model.encode(
            cleaned, normalize_embeddings=True, show_progress_bar=False
        )

    def scores(self, query: str) -> np.ndarray:
        q_emb = self.model.encode(
            [light_clean(query)], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return self.embeddings @ q_emb

    def search(self, query: str, top_k: int = 3):
        s = self.scores(query)
        top = np.argsort(-s)[:top_k]
        return [(int(i), float(s[i])) for i in top]
