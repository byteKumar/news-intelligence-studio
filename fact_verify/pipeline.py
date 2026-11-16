"""End-to-end claim extraction, retrieval, and verification pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .claim_extractor import extract_claims
from .loaders import build_raw_evidence_corpus
from .retriever import TfidfRetriever
from .summarizer import summarize_article
from .verifier import verify_claim


class FactVerificationPipeline:
    """Verification baseline that always builds evidence from the raw datasets."""

    def __init__(self, base_dir: str | Path, force_rebuild_corpus: bool = False, max_docs: int | None = 5000):
        """Initialize retriever with cached/built evidence corpus.

        Args:
            base_dir: Project root path.
            force_rebuild_corpus: If True, rebuild corpus cache from raw files.
            max_docs: Optional corpus cap for faster demo-time retrieval.
        """
        self.base_dir = Path(base_dir)
        corpus_df = build_raw_evidence_corpus(self.base_dir, force_rebuild=force_rebuild_corpus)
        self.corpus_df = corpus_df.head(max_docs).copy() if max_docs is not None else corpus_df
        self.retriever = TfidfRetriever(self.corpus_df)

    def analyze_article(self, headline: str, article_text: str, claims_top_k: int = 3, evidence_top_k: int = 3):
        """Run claim extraction, retrieval, and verdict generation.

        Returns:
            A serializable dict with:
            - headline: original headline text
            - summary: short article summary
            - claims: list of claim-level outputs (claim, claim_score, verdict, evidence)
            - corpus_size: number of indexed evidence documents
        """
        full_text = f"{headline.strip()} {article_text.strip()}".strip()
        claims = extract_claims(full_text, top_k=claims_top_k)
        outputs = []
        for claim in claims:
            hits = self.retriever.retrieve(claim.text, top_k=evidence_top_k)
            verdict = verify_claim(claim.text, hits)
            outputs.append({
                'claim': claim.text,
                'claim_score': claim.score,
                'verdict': asdict(verdict),
                'evidence': [asdict(hit) for hit in hits],
            })
        summary = summarize_article(full_text, max_sentences=3)
        return {
            'headline': headline,
            'summary': summary,
            'claims': outputs,
            'corpus_size': int(len(self.corpus_df)),
        }
