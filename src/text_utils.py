"""Utility helpers for cleaning and normalizing article text."""

from __future__ import annotations
import re
from typing import List

_SENT_RE = re.compile(r'(?<=[.!?])\s+')
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def normalize_text(text: str) -> str:
    """Aggressive normalization: lowercase, strip URLs, keep only a-z and spaces."""
    text = str(text or '').lower()
    text = re.sub(r"http\S+|www\.\S+", ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_light(text: str) -> str:
    """Light normalization: preserve case and punctuation, just collapse whitespace."""
    text = str(text or '').replace('\n', ' ').replace('\t', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text: str, min_words: int = 4) -> List[str]:
    """Split text on sentence boundaries.

    Args:
        text: Input text to split.
        min_words: Minimum word count for a sentence to be included. Use 0
            to keep all sentences regardless of length.
    """
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    if min_words > 0:
        return [s.strip() for s in parts if s.strip() and len(s.strip().split()) > min_words]
    return [s.strip() for s in parts if s.strip()]


def tokenize(text: str) -> List[str]:
    """Extract word tokens from text, lowercased."""
    return [w.lower() for w in _WORD_RE.findall(text or '')]
