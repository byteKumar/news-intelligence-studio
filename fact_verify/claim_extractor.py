"""Simple claim extraction heuristics for the verification demo."""

from dataclasses import dataclass
from typing import List

from .text_utils import split_sentences, tokenize


@dataclass
class Claim:
    text: str
    score: float


def _sentence_score(sentence: str) -> float:
    tokens = tokenize(sentence)
    if len(tokens) < 6:
        return 0.0
    score = 0.0
    score += min(len(tokens) / 18.0, 2.0)
    if any(k in sentence.lower() for k in [" is ", " was ", " were ", " has ", " have ", " said ", " according to "]):
        score += 0.6
    if any(ch.isdigit() for ch in sentence):
        score += 0.25
    if sentence.endswith("?"):
        score -= 0.5
    return score


def extract_claims(article_text: str, top_k: int = 3) -> List[Claim]:
    sentences = split_sentences(article_text)
    scored = [Claim(text=s, score=_sentence_score(s)) for s in sentences]
    scored = [c for c in scored if c.score > 0]
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]
