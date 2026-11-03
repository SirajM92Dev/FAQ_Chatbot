"""FAQ chatbot with pluggable retriever and a confidence-based fallback."""
import pandas as pd

from .tfidf_retriever import TfidfRetriever

FALLBACK_MESSAGE = (
    "Sorry, I couldn't find a relevant answer to that. "
    "Please rephrase your question or contact a qualified lawyer."
)

# Score scales differ per method, so each gets its own threshold.
# tfidf / wordvec / hybrid values come from the sweep on the bundled eval set;
# "semantic" is an untuned starting point: re-run evaluate.py with sentence-transformers.
DEFAULT_THRESHOLDS = {"tfidf": 0.02, "semantic": 0.45, "wordvec": 0.60, "hybrid": 0.30}


class FAQBot:
    def __init__(self, faq_path="data/faq.csv", method="semantic", threshold=None, extra_texts=()):
        self.faq = pd.read_csv(faq_path)
        questions = self.faq["question"].tolist()

        if method == "tfidf":
            self.retriever = TfidfRetriever(questions)
        elif method == "semantic":
            from .semantic_retriever import SemanticRetriever
            self.retriever = SemanticRetriever(questions)
        elif method == "wordvec":
            from .wordvec_retriever import WordVecRetriever
            self.retriever = WordVecRetriever(questions, extra_texts=extra_texts)
        elif method == "hybrid":
            from .semantic_retriever import SemanticRetriever
            from .hybrid_retriever import HybridRetriever
            self.retriever = HybridRetriever(
                TfidfRetriever(questions), SemanticRetriever(questions)
            )
        else:
            raise ValueError("method must be 'tfidf', 'semantic', 'wordvec' or 'hybrid'")

        self.method = method
        self.threshold = DEFAULT_THRESHOLDS[method] if threshold is None else threshold

    def ask(self, query: str, top_k: int = 3) -> dict:
        results = self.retriever.search(query, top_k=top_k)
        best_idx, best_score = results[0]

        if best_score < self.threshold:
            return {"answer": FALLBACK_MESSAGE, "matched_question": None,
                    "matched_id": -1, "score": best_score,
                    "fallback": True, "top_k": results}

        row = self.faq.iloc[best_idx]
        return {"answer": row["answer"], "matched_question": row["question"],
                "matched_id": int(row["id"]), "score": best_score,
                "fallback": False, "top_k": results}
