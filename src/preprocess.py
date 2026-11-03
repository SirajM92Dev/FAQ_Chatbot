"""Text preprocessing utilities.

Two levels of cleaning are used on purpose:
- heavy_clean: for keyword methods (TF-IDF). Removes stopwords, lemmatizes.
- light_clean: for transformer embeddings. Keeps word order and stopwords,
  because sentence transformers use full context.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def _ensure_nltk():
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()
_STOP = set(stopwords.words("english"))
_LEMMA = WordNetLemmatizer()


def light_clean(text: str) -> str:
    """Lowercase, strip extra whitespace and odd characters."""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s\?\.,']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def heavy_clean(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, drop stopwords, lemmatize."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOP and len(t) > 1]
    tokens = [_LEMMA.lemmatize(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    q = "How do I file an FIR against someone who stole my phones?"
    print("light:", light_clean(q))
    print("heavy:", heavy_clean(q))
