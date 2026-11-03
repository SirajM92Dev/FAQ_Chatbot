# Intelligent FAQ Chatbot: Keyword vs Semantic Retrieval

An FAQ chatbot over a Pakistani legal knowledge base, built to **compare how
different retrieval techniques understand and match user questions**. The chatbot
is a thin layer; the focus is the experiment: TF-IDF + cosine similarity versus
embedding-based retrieval, with a confidence-based fallback and a proper evaluation.

## Research question

When users paraphrase a question with different words, how much better is
embedding-based search than keyword TF-IDF, and what does each method get wrong?

## Pipeline

1. **Dataset**: `data/faq.csv` (30 question-answer pairs) and `data/eval_queries.csv`
   (54 labelled queries: 6 exact, 30 paraphrase, 10 keyword, 8 out-of-scope).
2. **Preprocessing** (`src/preprocess.py`): lowercasing, punctuation removal,
   tokenization, stopword removal and lemmatization for TF-IDF and word vectors;
   only light cleaning for Sentence Transformers, which need full context.
3. **Keyword retrieval** (`src/tfidf_retriever.py`): TF-IDF (unigrams + bigrams).
4. **Semantic retrieval** (`src/semantic_retriever.py`): `all-MiniLM-L6-v2`
   Sentence Transformer embeddings, normalized cosine similarity.
5. **Word-vector baseline** (`src/wordvec_retriever.py`): averaged GloVe vectors,
   a static (context-free) embedding model used as a middle point between TF-IDF
   and Sentence Transformers.
6. **Hybrid** (`src/hybrid_retriever.py`): 0.5 * dense score + 0.5 * TF-IDF score.
7. **Fallback** (`src/chatbot.py`): if the top score is below a per-method threshold,
   the bot declines instead of guessing.
8. **Evaluation** (`src/evaluate.py`, `src/compare.py`): Hit@1, Hit@3, MRR, results
   by query type, fallback precision/recall, threshold sweep, side-by-side errors.

## Setup

    pip install -r requirements.txt

Optional, for the word-vector baseline (about 130 MB): download
`glove-wiki-gigaword-100.gz` from the gensim-data releases page
(https://github.com/piskvorky/gensim-data/releases/tag/glove-wiki-gigaword-100)
and save it as `data/glove-wiki-gigaword-100.gz`. It is git-ignored.

## Run

    python -m src.evaluate        # metrics + threshold sweep -> results/
    python -m src.compare         # side-by-side errors -> results/side_by_side.csv
    python app.py --method both   # interactive chatbot (tfidf + semantic)
    python app.py --method hybrid

`evaluate.py` runs every method whose dependencies are available and skips the rest.

## Results

Measured on the bundled eval set (46 in-scope queries, 8 out-of-scope):

| Method  | Hit@1 | Hit@3 | MRR   | Hit@1 exact | Hit@1 paraphrase | Hit@1 keyword | Fallback P | Fallback R | System acc. |
|---------|-------|-------|-------|-------------|------------------|---------------|------------|------------|-------------|
| TF-IDF  | 0.391 | 0.435 | 0.456 | 1.00        | 0.133            | 0.80          | 0.381      | 1.00       | 0.481       |
| GloVe   | 0.435 | 0.609 | 0.552 | 1.00        | 0.200            | 0.80          | 0.750      | 0.75       | 0.481       |
| Hybrid  | 0.457 | 0.609 | 0.564 | 1.00        | 0.233            | 0.80          | 0.750      | 0.75       | 0.500       |

These results were produced with TF-IDF, the GloVe word-vector baseline and the
hybrid of the two. The Sentence Transformer retriever is implemented
(`src/semantic_retriever.py`) but was not evaluated in the environment used for
these numbers, because the model could not be downloaded there. Running
`python -m src.evaluate` with internet access adds a `semantic` row and a
semantic-based hybrid automatically.

## Analysis

- **Exact and keyword queries are easy for every method.** All hit 100% on exact
  questions and 80% on single-keyword queries. Plain keyword matching works well
  when the user reuses the FAQ wording.
- **Paraphrases separate the methods.** TF-IDF gets 13% Hit@1 because the test
  paraphrases share few content words with the FAQ question. With no overlap its
  score is zero and the top result is arbitrary: it returned "How do I file an FIR?"
  (the first FAQ) for "protect my brand name and logo legally" and for "seller sold
  me a defective product". GloVe fixed four such cases by relating *brand/trademark*,
  *defective product/consumer complaint*, *talaq/divorce* and *court decision/judgment*.
- **Word vectors are not enough.** GloVe still reaches only 20% Hit@1 on paraphrases.
  Averaged static vectors blur everything that mentions "court" or "police": it
  matched "court order pausing an eviction" to *copy of a court judgment*, and
  "how can I report a theft to the police" to *police misconduct* instead of *FIR*.
  GloVe also lost two cases TF-IDF got right, for example "do I need a lawyer or
  can I go to court alone".
- **Rare terms are a vocabulary problem.** "nikahnama" is out-of-vocabulary for
  GloVe, and "will" is a stopword, so both queries failed for every method.
- **TF-IDF gives a clean fallback signal but cannot separate "unknown" from
  "paraphrased".** Out-of-scope queries score 0.0, which is easy to detect, but
  most paraphrases also score near 0, so any threshold that rejects out-of-scope
  queries also rejects valid ones (fallback precision 0.381).
- **Dense scores are less calibrated.** GloVe gave out-of-scope queries high scores
  (0.68 for the weather question, 0.66 for the gold price question) because averaged
  vectors are similar to each other. Fallback precision rose to 0.75 but recall fell
  to 0.75.
- **Hybrid helps a little.** It has the best Hit@1 (0.457) and system accuracy
  (0.50), but the gain over GloVe alone is small on this dataset.

Full per-query results are produced in `results/side_by_side.csv`.

## Limitations

- Small dataset (30 FAQs, 54 queries). Differences of a few points are within noise.
- Paraphrases were written to avoid word overlap, so the set is deliberately hard
  and favours embeddings. Real user queries will overlap more.
- Thresholds were chosen on the same eval set used for reporting, so the fallback
  numbers are optimistic. A stricter study would split tuning and test sets.
- The Sentence Transformer was not evaluated (see Results).
- Answers are sample content, not legal advice. Verify before real use.

## Future work

- Evaluate `all-MiniLM-L6-v2`, `multi-qa-MiniLM-L6-cos-v1`, `bge-small` and a
  multilingual model for Urdu queries.
- Cross-encoder re-ranking of the top 3 results.
- RAG: pass the top-k FAQ answers to an LLM and evaluate faithfulness.
- Grow the FAQ set to 100+ entries and split the eval set into dev and test.

## Tech stack

Python, scikit-learn, NLTK, sentence-transformers, GloVe, pandas, matplotlib
