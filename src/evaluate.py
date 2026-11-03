"""Evaluate retrieval quality: TF-IDF vs embedding-based retrieval.

Ranking metrics (in-scope queries): Hit@1, Hit@3, MRR, Hit@1 by query type.
System metrics (all queries): accuracy with fallback, fallback precision/recall.

Methods run automatically if their dependencies are available:
  tfidf     always
  semantic  needs sentence-transformers + model download (Hugging Face)
  wordvec   needs data/glove-wiki-gigaword-100.gz (see wordvec_retriever.py)
  hybrid    needs tfidf + semantic (or wordvec when semantic is unavailable)

Run:  python -m src.evaluate
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .tfidf_retriever import TfidfRetriever
from .hybrid_retriever import HybridRetriever

FAQ_PATH = "data/faq.csv"
EVAL_PATH = "data/eval_queries.csv"


def build_retrievers(faq, evals):
    questions = faq["question"].tolist()
    r = {"tfidf": TfidfRetriever(questions)}
    try:
        from .semantic_retriever import SemanticRetriever
        r["semantic"] = SemanticRetriever(questions)
    except Exception as e:  # no package / no model download
        print(f"[skip] semantic (sentence-transformers) unavailable: {type(e).__name__}")
    try:
        from .wordvec_retriever import WordVecRetriever
        r["wordvec"] = WordVecRetriever(questions, extra_texts=evals["query"].tolist())
    except Exception as e:
        print(f"[skip] wordvec unavailable: {type(e).__name__}")
    dense = r.get("semantic") or r.get("wordvec")
    if dense is not None:
        r["hybrid"] = HybridRetriever(r["tfidf"], dense, alpha=0.5)
    return r


def rank_info(retriever, faq, query):
    s = retriever.scores(query)
    order = np.argsort(-s)
    return [int(faq.iloc[i]["id"]) for i in order], [float(s[i]) for i in order]


def retrieval_metrics(retriever, faq, evals):
    scoped = evals[evals["true_id"] != -1]
    hit1, hit3, rr = [], [], []
    for _, row in scoped.iterrows():
        ids, _ = rank_info(retriever, faq, row["query"])
        rank = ids.index(int(row["true_id"])) + 1
        hit1.append(rank == 1); hit3.append(rank <= 3); rr.append(1.0 / rank)
    return {"Hit@1": np.mean(hit1), "Hit@3": np.mean(hit3), "MRR": np.mean(rr)}


def by_type(retriever, faq, evals):
    out = {}
    for t in ["exact", "paraphrase", "keyword"]:
        sub = evals[evals["type"] == t]
        hits = [rank_info(retriever, faq, r["query"])[0][0] == int(r["true_id"])
                for _, r in sub.iterrows()]
        out[t] = float(np.mean(hits)) if hits else float("nan")
    return out


def fallback_metrics(retriever, faq, evals, threshold):
    """Full system (retrieval + fallback). Positive class = 'bot says fallback'."""
    correct = tp = fp = fn = 0
    for _, row in evals.iterrows():
        ids, scores = rank_info(retriever, faq, row["query"])
        pred = -1 if scores[0] < threshold else ids[0]
        true = int(row["true_id"])
        correct += int(pred == true)
        tp += int(true == -1 and pred == -1)
        fp += int(true != -1 and pred == -1)
        fn += int(true == -1 and pred != -1)
    return {
        "system_accuracy": correct / len(evals),
        "fallback_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "fallback_recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def threshold_sweep(retriever, faq, evals, grid):
    rows = []
    for th in grid:
        m = fallback_metrics(retriever, faq, evals, float(th)); m["threshold"] = float(th)
        rows.append(m)
    return pd.DataFrame(rows)


def best_threshold(sweep):
    """Highest system accuracy; ties -> highest fallback recall, then lowest threshold."""
    s = sweep.sort_values(["system_accuracy", "fallback_recall", "threshold"],
                          ascending=[False, False, True])
    return float(s.iloc[0]["threshold"])


def main():
    faq = pd.read_csv(FAQ_PATH)
    evals = pd.read_csv(EVAL_PATH)
    retrievers = build_retrievers(faq, evals)

    rows, sweeps, chosen = [], {}, {}
    for name, r in retrievers.items():
        sweeps[name] = threshold_sweep(r, faq, evals, np.arange(0.0, 1.0, 0.02))
        chosen[name] = best_threshold(sweeps[name])
        sweeps[name].round(3).to_csv(f"results/threshold_sweep_{name}.csv", index=False)
        m = retrieval_metrics(r, faq, evals)
        t = by_type(r, faq, evals)
        f = fallback_metrics(r, faq, evals, chosen[name])
        rows.append({"method": name, **m,
                     **{f"Hit@1_{k}": v for k, v in t.items()},
                     "threshold": chosen[name], **f})

    summary = pd.DataFrame(rows).round(3)
    print("\n=== Retrieval comparison ===")
    print(summary.set_index("method").T.to_string())
    summary.to_csv("results/summary.csv", index=False)

    fig, axes = plt.subplots(1, len(retrievers), figsize=(4.5 * len(retrievers), 3.8), squeeze=False)
    for ax, name in zip(axes[0], retrievers):
        sw = sweeps[name]
        ax.plot(sw["threshold"], sw["system_accuracy"], label="System accuracy")
        ax.plot(sw["threshold"], sw["fallback_precision"], label="Fallback precision")
        ax.plot(sw["threshold"], sw["fallback_recall"], label="Fallback recall")
        ax.axvline(chosen[name], linestyle="--", color="gray")
        ax.set_title(f"{name}: threshold sweep"); ax.set_xlabel("similarity threshold")
        ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig("results/threshold_sweep.png", dpi=150)
    print("\nSaved results/summary.csv, threshold_sweep_*.csv, threshold_sweep.png")


if __name__ == "__main__":
    main()
