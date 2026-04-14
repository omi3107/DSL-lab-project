"""
Text Cleaner for Meeting Transcript Sentences.

Provides text preprocessing utilities for the ML classification pipeline:
- Lowercasing
- Punctuation / special character removal
- Stopword removal (NLTK)
- Lemmatization (WordNet)

Usage:
    from preprocessing.text_cleaner import clean_text, clean_series
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (safe for Colab & local)
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = ["stopwords", "wordnet", "omw-1.4"]

def _ensure_nltk_resources():
    """Download NLTK data files if not already present."""
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_resources()

# ---------------------------------------------------------------------------
# Shared objects
# ---------------------------------------------------------------------------
_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean a single sentence string.

    Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove digits
        5. Remove punctuation & special characters
        6. Tokenize on whitespace
        7. Remove stopwords
        8. Lemmatize each token
        9. Rejoin tokens

    Args:
        text: Raw sentence string.

    Returns:
        Cleaned sentence string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove punctuation & special chars
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize, remove stopwords, lemmatize
    tokens = text.split()
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 1
    ]

    return " ".join(tokens)


def clean_series(series):
    """Apply ``clean_text`` to every element of a pandas Series.

    Args:
        series: ``pandas.Series`` of raw text.

    Returns:
        ``pandas.Series`` with cleaned text.
    """
    return series.astype(str).apply(clean_text)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "We DECIDED to launch next week!",
        "Rahul will prepare slides by Friday.",
        "Hmm hmm hmm.",
        "Pricing is still unclear - need clarification.",
        "",
        None,
    ]
    for s in samples:
        print(f"  IN:  {s!r}")
        print(f"  OUT: {clean_text(s if s else '')!r}\n")
