"""Qualitative comparison: TF-IDF vs embedding retrieval, side by side.

Run:  python -m src.compare
Writes results/side_by_side.csv and prints queries where methods disagree.
"""
import pandas as pd

from .evaluate import build_retrievers, rank_info, FAQ_PATH, EVAL_PATH


def main():
    faq = pd.read_csv(FAQ_PATH)
    evals = pd.read_csv(EVAL_PATH)
    id_to_q = dict(zip(faq["id"], faq["question"]))
    retrievers = build_retrievers(faq, evals)

    rows = []
    for _, row in evals.iterrows():
        true = int(row["true_id"])
        rec = {"query": row["query"], "type": row["type"],
               "true": id_to_q.get(true, "OUT OF SCOPE")}
        for name, r in retrievers.items():
            ids, scores = rank_info(r, faq, row["query"])
            rec[f"{name}_match"] = id_to_q[ids[0]]
            rec[f"{name}_score"] = round(scores[0], 3)
            rec[f"{name}_correct"] = ids[0] == true
            if true != -1:
                rec[f"{name}_rank"] = ids.index(true) + 1
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv("results/side_by_side.csv", index=False)

    scoped = df[df["true"] != "OUT OF SCOPE"]
    dense = "semantic" if "semantic" in retrievers else "wordvec"
    if dense in retrievers:
        wins = scoped[scoped[f"{dense}_correct"] & ~scoped["tfidf_correct"]]
        loss = scoped[scoped["tfidf_correct"] & ~scoped[f"{dense}_correct"]]
        both = scoped[~scoped[f"{dense}_correct"] & ~scoped["tfidf_correct"]]
        print(f"{dense} correct, TF-IDF wrong: {len(wins)}")
        print(wins[["query", "tfidf_match", f"{dense}_match"]].to_string(index=False))
        print(f"\nTF-IDF correct, {dense} wrong: {len(loss)}")
        print(loss[["query", "tfidf_match", f"{dense}_match"]].to_string(index=False))
        print(f"\nBoth wrong: {len(both)}")
    print("\nSaved results/side_by_side.csv")


if __name__ == "__main__":
    main()
