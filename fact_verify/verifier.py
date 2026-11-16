"""Lightweight rule-based claim verifier for the bundled demo corpus."""

from dataclasses import dataclass
from typing import List
from .text_utils import tokenize
from .retriever import EvidenceHit

NEGATIONS = {"not", "no", "never", "without", "none", "n't"}


@dataclass
class VerificationResult:
    """Final verdict object returned for one claim."""
    label: str
    confidence: float
    rationale: str


def _negation_present(tokens: list[str]) -> bool:
    """Detect if a token list contains negation cues."""
    return any(tok in NEGATIONS or tok.endswith("n't") for tok in tokens)


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    """Compute coverage of claim tokens found in evidence tokens."""
    if not a:
        return 0.0
    return len(a & b) / max(len(a), 1)


def verify_claim(claim_text: str, evidence_hits: List[EvidenceHit]) -> VerificationResult:
    """Assign SUPPORTED/REFUTED/NOT ENOUGH EVIDENCE for a single claim.

    Scoring heuristic:
    - base score = 0.65 * lexical overlap + 0.35 * retriever similarity
    - optional +0.08 for contradiction signal when negation polarity differs
      and overlap is sufficiently strong.

    Decision thresholds:
    - REFUTED if best contradiction score >= 0.42 and clearly above support
    - SUPPORTED if best support score >= 0.40
    - otherwise NOT ENOUGH EVIDENCE
    """
    claim_tokens = [t for t in tokenize(claim_text) if len(t) > 2]
    claim_set = set(claim_tokens)
    claim_neg = _negation_present(claim_tokens)

    best_support = 0.0
    best_refute = 0.0
    best_hit = None

    for hit in evidence_hits:
        e_tokens = [t for t in tokenize(hit.passage) if len(t) > 2]
        e_set = set(e_tokens)
        overlap = _overlap_ratio(claim_set, e_set)
        evidence_neg = _negation_present(e_tokens)
        # Blend token-level claim coverage with semantic retrieval strength.
        score = 0.65 * overlap + 0.35 * hit.score
        if evidence_neg != claim_neg and overlap > 0.25:
            # Negation mismatch can indicate contradiction/refutation.
            score += 0.08
            if score > best_refute:
                best_refute = score
                best_hit = hit
        else:
            if score > best_support:
                best_support = score
                best_hit = hit

    if best_refute >= 0.42 and best_refute > best_support + 0.05:
        return VerificationResult(
            label="REFUTED",
            confidence=min(best_refute, 0.95),
            rationale=f"Top evidence conflicts with the claim wording and shows strong token overlap with '{best_hit.title}'.",
        )
    if best_support >= 0.40:
        return VerificationResult(
            label="SUPPORTED",
            confidence=min(best_support, 0.95),
            rationale=f"Top evidence strongly overlaps with the claim and is consistent with '{best_hit.title}'.",
        )
    return VerificationResult(
        label="NOT ENOUGH EVIDENCE",
        confidence=max(best_support, best_refute, 0.22),
        rationale="Retrieved evidence is too weak or incomplete to confidently support or refute the claim.",
    )
