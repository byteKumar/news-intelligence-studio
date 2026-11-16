"""Text utilities for the fact-verification module.

Tokenization is shared from src.text_utils. Light normalization and
no-minimum-word sentence splitting are kept here because the verification
pipeline intentionally preserves punctuation and short sentences that the
aggressive src normalizer would strip.
"""

from __future__ import annotations
import re
from typing import List

from src.text_utils import tokenize  # shared implementation, no duplication

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def normalize_text(text: str) -> str:
    """Light normalization: preserve case and punctuation, collapse whitespace."""
    text = str(text or '').replace('\n', ' ').replace('\t', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences with no minimum word-count filter."""
    text = normalize_text(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


__all__ = ['normalize_text', 'split_sentences', 'tokenize']
