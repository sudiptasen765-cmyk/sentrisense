"""
ml/preprocessing/clean_text.py

Text cleaning for SentriSense.

Design principle: cleaning must never destroy sentiment-bearing signal.
Concretely this means:
  - Never strip stopwords. Words like "not", "never", "no", "n't" flip
    sentiment meaning ("not good" != "good") and a generic stopword list
    would delete them.
  - Only remove things that are genuinely noise: HTML artifacts, excess
    whitespace, non-printable characters.
  - Two cleaning modes exist because classical TF-IDF models and
    Transformer models want different amounts of normalization:
      * clean_for_classical(): lowercases, strips HTML, normalizes
        whitespace. Punctuation is left in place — scikit-learn's default
        TF-IDF tokenizer already ignores punctuation via its token pattern,
        so stripping it here would be redundant, not helpful.
      * clean_for_transformer(): strips HTML and normalizes whitespace only.
        Case and punctuation are left untouched, since subword tokenizers
        (e.g. DistilBERT's) are trained on natural, cased, punctuated text
        and lowercasing would throw away information the model expects.
"""

import html
import re

# Compiled once at import time for speed across 25k+ calls.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_PUNCT_RE = re.compile(r"([!?.]){3,}")  # "!!!!!!" -> "!!!" (keeps emphasis signal, caps spam)

# Used only for documentation/testing — proof that these survive cleaning,
# NOT used to filter or remove anything.
NEGATION_PROBE_WORDS = ("not", "never", "no", "n't", "cannot", "neither", "nor")


def remove_html(text: str) -> str:
    """Strip HTML tags (IMDb reviews contain literal '<br />' line breaks)
    and unescape HTML entities (e.g. '&amp;' -> '&')."""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines/tabs) into single spaces
    and trim leading/trailing whitespace."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def cap_repeated_punctuation(text: str) -> str:
    """Cap runs of 3+ '!'/'?'/'.' down to 3, preserving the emphasis signal
    ('amazing!!!!!!' still reads as emphatic) without letting outliers skew
    any length-based features downstream."""
    return _MULTI_PUNCT_RE.sub(r"\1\1\1", text)


def clean_for_classical(text: str) -> str:
    """Cleaning pipeline for TF-IDF + Logistic Regression / Linear SVM."""
    if not isinstance(text, str):
        return ""
    text = remove_html(text)
    text = text.lower()
    text = cap_repeated_punctuation(text)
    text = normalize_whitespace(text)
    return text


def clean_for_transformer(text: str) -> str:
    """Cleaning pipeline for the DistilBERT comparison model. Deliberately
    minimal — case and punctuation carry information subword tokenizers use."""
    if not isinstance(text, str):
        return ""
    text = remove_html(text)
    text = cap_repeated_punctuation(text)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    # Quick manual sanity check when run directly (automated checks live in
    # test_clean_text.py — this is just for eyeballing during development).
    samples = [
        "This movie was <br /><br />NOT good at all.",
        "I never liked the acting, but the plot was decent.",
        "Honestly? Not terrible. Wouldn't watch again though.",
        "AMAZING!!!!!!!!! Best film ever &amp; then some.",
    ]
    for s in samples:
        print(f"RAW:        {s!r}")
        print(f"CLASSICAL:  {clean_for_classical(s)!r}")
        print(f"TRANSFORMER:{clean_for_transformer(s)!r}")
        print()
