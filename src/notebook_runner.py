"""Utilities used by the notebook-first execution path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fact_verify.pipeline import FactVerificationPipeline
from src.pipeline import predict_article
from src.train_pipeline import build_project_assets


def run_full_rebuild(base_dir: str | Path, n_topics: int = 12, extra_real_rows: int = 2000, topic_rows: int = 12000) -> dict[str, Any]:
    """Train the project from raw data using the full labeled classifier set and a large stratified topic sample."""
    base_dir = Path(base_dir)
    summary = build_project_assets(
        base_dir=base_dir,
        classifier_rows=None,
        extra_real_rows=extra_real_rows,
        topic_rows=topic_rows,
        n_topics=n_topics,
    )
    return summary


def load_saved_outputs(base_dir: str | Path) -> dict[str, Any]:
    base_dir = Path(base_dir)
    outputs: dict[str, Any] = {}
    outputs['build_summary'] = json.loads((base_dir / 'results' / 'build_summary.json').read_text())
    outputs['classification_metrics'] = json.loads((base_dir / 'results' / 'classification_metrics.json').read_text())
    outputs['classifier_benchmark'] = pd.read_csv(base_dir / 'results' / 'model_benchmark.csv')
    outputs['topic_benchmark'] = pd.read_csv(base_dir / 'results' / 'topic_model_benchmark.csv')
    outputs['source_breakdown'] = pd.read_csv(base_dir / 'results' / 'source_breakdown.csv')
    outputs['dataset_split'] = pd.read_csv(base_dir / 'results' / 'classification_split_table.csv')
    return outputs


def analyze_and_save_custom_article(base_dir: str | Path, headline: str, article_text: str) -> dict[str, Any]:
    base_dir = Path(base_dir)
    analysis_result = predict_article(headline, article_text, base_dir=base_dir)
    verifier = FactVerificationPipeline(base_dir, max_docs=5000)
    verification_result = verifier.analyze_article(headline, article_text)
    saved = {
        'analysis': analysis_result,
        'verification': verification_result,
    }
    (base_dir / 'results' / 'custom_article_result.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
    return saved
