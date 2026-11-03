"""Optional offline embedding baseline: averaged GloVe word vectors.

Not a sentence transformer. It is a static word-embedding model (each word has one
fixed vector, no context). Useful as a middle point between TF-IDF and
Sentence Transformers, and runs without downloading from Hugging Face.

Get the vectors (gensim-data release, ~130 MB):
  https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
Save as data/glove-wiki-gigaword-100.gz (git-ignored) or set GLOVE_PATH.
"""
import gzip
import os

import numpy as np

from .preprocess import heavy_clean

GLOVE_PATH = os.environ.get("GLOVE_PATH", "data/glove-wiki-gigaword-100.gz")


def load_glove(path, vocab):
    """Load vectors. vocab=None loads every word; otherwise only words in vocab."""
    vecs = {}
    with gzip.open(path, "rt", encoding="utf8") as f:
        next(f)  # header line: "<n_words> <dim>"
        for line in f:
            word, rest = line.split(" ", 1)
            if vocab is None or word in vocab:
                vecs[word] = np.asarray(rest.split(), dtype=np.float32)
    return vecs


class WordVecRetriever:
    name = "wordvec"

    def __init__(self, questions, path=GLOVE_PATH, extra_texts=()):
        """extra_texts: other texts (e.g. test queries) whose words need vectors."""
        self.questions = list(questions)
        # With no extra_texts (interactive use) load the whole file so any
        # query word has a vector; with extra_texts load only what is needed.
        vocab = None
        if extra_texts:
            vocab = set()
            for t in extra_texts:
                vocab.update(heavy_clean(t).split())
        for q in self.questions:
            if vocab is not None:
                vocab.update(heavy_clean(q).split())
        self.path = path
        self.vecs = load_glove(path, vocab)
        self.dim = len(next(iter(self.vecs.values())))
        self.matrix = np.vstack([self._embed(q) for q in self.questions])

    def _embed(self, text):
        toks = [t for t in heavy_clean(text).split() if t in self.vecs]
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        v = np.mean([self.vecs[t] for t in toks], axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def scores(self, query):
        return self.matrix @ self._embed(query)

    def search(self, query, top_k=3):
        s = self.scores(query)
        top = np.argsort(-s)[:top_k]
        return [(int(i), float(s[i])) for i in top]
