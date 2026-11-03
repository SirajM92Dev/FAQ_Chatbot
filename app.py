"""Command-line chatbot.

Usage:
    python app.py --method semantic
    python app.py --method tfidf
    python app.py --method both     (tfidf + semantic side by side)
    python app.py --method hybrid
"""
import argparse

from src.chatbot import FAQBot


def show(bot, label, query):
    r = bot.ask(query)
    tag = "FALLBACK" if r["fallback"] else "MATCH"
    print(f"\n[{label}] {tag} (score={r['score']:.3f})")
    if not r["fallback"]:
        print(f"  Matched: {r['matched_question']}")
    print(f"  Answer : {r['answer']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tfidf", "semantic", "hybrid", "both"], default="semantic")
    args = parser.parse_args()

    bots = {}
    if args.method in ("tfidf", "both"):
        bots["TF-IDF"] = FAQBot(method="tfidf")
    if args.method in ("semantic", "both"):
        bots["Semantic"] = FAQBot(method="semantic")
    if args.method == "hybrid":
        bots["Hybrid"] = FAQBot(method="hybrid")

    print("FAQ Chatbot ready. Type 'quit' to exit.")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue
        for label, bot in bots.items():
            show(bot, label, query)


if __name__ == "__main__":
    main()
```

## File: results/.gitkeep

```
