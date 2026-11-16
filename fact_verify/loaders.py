"""Build verification corpora directly from the raw datasets."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data_loader import load_kaggle_fake_real, load_fakenewsnet, load_news_category
from src.text_utils import normalize_text

REQUIRED_CORPUS_COLUMNS = {"doc_id", "title", "passage", "source"}


def _passage_from_text(text: str, max_chars: int = 700) -> str:
    cleaned = normalize_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(' ', 1)[0] + '…'


def build_raw_evidence_corpus(base_dir: str | Path, force_rebuild: bool = False) -> pd.DataFrame:
    """Create a retrieval corpus from the raw project datasets and cache it."""
    base_dir = Path(base_dir)
    cache_path = base_dir / 'data' / 'processed' / 'evidence_corpus_from_raw.csv'
    if cache_path.exists() and not force_rebuild:
        cached = pd.read_csv(cache_path)
        missing = REQUIRED_CORPUS_COLUMNS - set(cached.columns)
        if not missing:
            return cached

    kaggle = load_kaggle_fake_real(base_dir)
    fakenewsnet = load_fakenewsnet(base_dir)
    news_category = load_news_category(base_dir)

    evidence_sources = []
    for source_name, df in [('kaggle', kaggle[kaggle['label'] == 'REAL']), ('fakenewsnet', fakenewsnet[fakenewsnet['label'] == 'REAL']), ('news_category', news_category)]:
        frame = df.copy()
        frame['passage'] = frame['text'].fillna('').astype(str).map(_passage_from_text)
        frame['title'] = frame['title'].fillna('').astype(str)
        frame = frame[(frame['title'].str.len() > 5) & (frame['passage'].str.len() > 40)].copy()
        frame['source'] = source_name
        evidence_sources.append(frame[['title', 'passage', 'source']])

    corpus = pd.concat(evidence_sources, ignore_index=True).drop_duplicates(subset=['title', 'passage']).reset_index(drop=True)
    corpus.insert(0, 'doc_id', [f'doc_{idx:06d}' for idx in range(1, len(corpus) + 1)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(cache_path, index=False)
    return corpus


def build_sample_articles_from_raw(base_dir: str | Path, force_rebuild: bool = False) -> pd.DataFrame:
    """Create a small verification sample set directly from the raw article files."""
    base_dir = Path(base_dir)
    cache_path = base_dir / 'data' / 'processed' / 'verification_sample_articles.csv'
    if cache_path.exists() and not force_rebuild:
        return pd.read_csv(cache_path)

    kaggle = load_kaggle_fake_real(base_dir)
    sample = (
        kaggle[kaggle['text'].fillna('').str.len() > 120]
        .groupby('subject', group_keys=False)
        .head(3)
        .copy()
    )
    sample = sample.rename(columns={'title': 'headline', 'text': 'article_text'})[['headline', 'article_text', 'label', 'subject']].reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(cache_path, index=False)
    return sample
