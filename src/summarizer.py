"""Extractive summarization helpers used by the article analyzer."""

from __future__ import annotations
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from .text_utils import split_sentences, tokenize

STOPWORDS = set(CountVectorizer(stop_words='english').get_stop_words())


def summarize(text: str, max_sentences: int = 3) -> str:
    """TF-frequency summarization with a position bias for earlier sentences.

    Sentences are scored by the sum of normalized word frequencies plus a
    small bonus that decays with sentence index, then reordered by original
    position before joining.
    """
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return ' '.join(sentences)
    words = re.findall(r'[a-z]+', str(text).lower())
    frequencies: dict[str, float] = {}
    for word in words:
        if word not in STOPWORDS and len(word) > 2:
            frequencies[word] = frequencies.get(word, 0) + 1
    if not frequencies:
        return ' '.join(sentences[:max_sentences])
    max_freq = max(frequencies.values())
    frequencies = {w: v / max_freq for w, v in frequencies.items()}
    scored = []
    for idx, sentence in enumerate(sentences):
        sent_words = re.findall(r'[a-z]+', sentence.lower())
        score = sum(frequencies.get(w, 0) for w in sent_words if w not in STOPWORDS)
        score += max(0, 0.18 - 0.04 * idx)
        scored.append((idx, score, sentence))
    best = sorted(scored, key=lambda row: row[1], reverse=True)[:max_sentences]
    best = sorted(best, key=lambda row: row[0])
    return ' '.join(sentence for _, _, sentence in best)


def summarize_basic(text: str, max_sentences: int = 3) -> str:
    """Pure frequency-based summarization without position bias.

    Each sentence is scored by the mean token frequency across its tokens.
    Used by the fact-verification pipeline where sentence order matters less
    than surfacing the most information-dense content.
    """
    sentences = split_sentences(text, min_words=0)
    if len(sentences) <= max_sentences:
        return ' '.join(sentences)
    freqs = Counter(tokenize(text))
    scored = []
    for sent in sentences:
        toks = tokenize(sent)
        if not toks:
            continue
        score = sum(freqs[t] for t in toks) / len(toks)
        scored.append((score, sent))
    chosen = [s for _, s in sorted(scored, reverse=True)[:max_sentences]]
    return ' '.join(chosen)
