"""Traditional keyword-based retrieval: TF-IDF + cosine similarity."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import heavy_clean


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, questions, ngram_range=(1, 2)):
        self.questions = list(questions)
        self.vectorizer = TfidfVectorizer(
            preprocessor=heavy_clean, ngram_range=ngram_range
        )
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def scores(self, query: str) -> np.ndarray:
        q_vec = self.vectorizer.transform([query])
        return cosine_similarity(q_vec, self.matrix).ravel()

    def search(self, query: str, top_k: int = 3):
        """Return list of (index, score) sorted by score descending."""
        s = self.scores(query)
        top = np.argsort(-s)[:top_k]
        return [(int(i), float(s[i])) for i in top]
