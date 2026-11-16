from pathlib import Path

from fact_verify.claim_extractor import extract_claims
from fact_verify.pipeline import FactVerificationPipeline
from fact_verify.retriever import TfidfRetriever
from fact_verify.verifier import verify_claim
from fact_verify.loaders import build_raw_evidence_corpus, build_sample_articles_from_raw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_claim_extractor_finds_scored_claims() -> None:
    claims = extract_claims(
        'NASA said the mission launched successfully in 2024. Observers later confirmed the landing.',
        top_k=2,
    )
    assert len(claims) >= 1
    assert claims[0].score > 0


def test_retrieval_and_verification_pipeline_runs() -> None:
    corpus_df = build_raw_evidence_corpus(PROJECT_ROOT)
    retriever = TfidfRetriever(corpus_df.head(500))
    hits = retriever.retrieve('The World Health Organization declared covid-19 a pandemic in 2020.', top_k=2)
    verdict = verify_claim('The World Health Organization declared covid-19 a pandemic in 2020.', hits)
    assert len(hits) == 2
    assert verdict.label in {'SUPPORTED', 'REFUTED', 'NOT ENOUGH EVIDENCE'}


def test_fact_verification_pipeline_returns_claims() -> None:
    samples = build_sample_articles_from_raw(PROJECT_ROOT)
    row = samples.iloc[0]
    pipeline = FactVerificationPipeline(PROJECT_ROOT, max_docs=500)
    result = pipeline.analyze_article(
        headline=row['headline'],
        article_text=row['article_text'],
        claims_top_k=1,
        evidence_top_k=2,
    )
    assert 'claims' in result
    assert len(result['claims']) >= 1
