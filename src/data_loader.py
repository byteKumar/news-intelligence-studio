"""Dataset loading and sampling utilities for classifier/topic training.

Raw dataset loaders cache their results so that multiple callers (classifier
training, topic training, profiling, evidence corpus building) read each CSV
or JSON file from disk only once per process.  Public functions return a copy
so callers can safely modify the returned DataFrame.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pandas as pd


def stratified_sample(df: pd.DataFrame, column: str, n_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    """Return a stratified sample, or the full frame when ``n_rows`` is ``None``."""
    if n_rows is None or n_rows <= 0 or len(df) <= n_rows:
        return df.copy().reset_index(drop=True)
    groups = []
    counts = df[column].value_counts(normalize=True)
    for value, frac in counts.items():
        subset = df[df[column] == value]
        take = max(1, int(round(frac * n_rows)))
        take = min(take, len(subset))
        groups.append(subset.sample(take, random_state=random_state))
    out = pd.concat(groups, ignore_index=True).drop_duplicates()
    if len(out) < n_rows:
        remainder = df.drop(out.index, errors='ignore') if len(out.index) else df
        if len(remainder) > 0:
            extra = remainder.sample(min(n_rows - len(out), len(remainder)), random_state=random_state)
            out = pd.concat([out, extra], ignore_index=True)
    return out.sample(min(n_rows, len(out)), random_state=random_state).reset_index(drop=True)


def balanced_label_sample(df: pd.DataFrame, n_rows: int | None, random_state: int = 42) -> pd.DataFrame:
    """Return a balanced fake/real sample, or the full frame when ``n_rows`` is ``None``."""
    if n_rows is None or n_rows <= 0 or len(df) <= n_rows:
        return df.copy().reset_index(drop=True)
    half = max(1, n_rows // 2)
    fake = df[df['label'] == 'FAKE']
    real = df[df['label'] == 'REAL']
    fake_take = min(len(fake), half)
    real_take = min(len(real), n_rows - fake_take)
    out = pd.concat([
        fake.sample(fake_take, random_state=random_state),
        real.sample(real_take, random_state=random_state),
    ], ignore_index=True)
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal cached loaders — read each file from disk at most once per process.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _load_kaggle_cached(base_dir_str: str) -> pd.DataFrame:
    base_dir = Path(base_dir_str)
    fake = pd.read_csv(base_dir / 'data' / 'raw' / 'kaggle_fake_real' / 'Fake.csv')
    real = pd.read_csv(base_dir / 'data' / 'raw' / 'kaggle_fake_real' / 'True.csv')
    fake['label'] = 'FAKE'; real['label'] = 'REAL'
    fake['dataset_source'] = 'kaggle_fake_real'; real['dataset_source'] = 'kaggle_fake_real'
    for df in (fake, real):
        df['title'] = df['title'].fillna('').astype(str)
        df['text'] = df['text'].fillna('').astype(str)
        df['subject'] = df.get('subject', '').fillna('').astype(str)
        df['date'] = df.get('date', '').fillna('').astype(str)
        df['source_group'] = df['subject']
    return pd.concat([fake, real], ignore_index=True)[['title', 'text', 'subject', 'date', 'label', 'dataset_source', 'source_group']]


@lru_cache(maxsize=8)
def _load_fakenewsnet_cached(base_dir_str: str) -> pd.DataFrame:
    root = Path(base_dir_str) / 'data' / 'raw' / 'fakenewsnet'
    frames = []
    for name in ['BuzzFeed_fake_news_content.csv', 'BuzzFeed_real_news_content.csv',
                 'PolitiFact_fake_news_content.csv', 'PolitiFact_real_news_content.csv']:
        df = pd.read_csv(root / name)
        df['title'] = df['title'].fillna('').astype(str)
        df['text'] = df['text'].fillna('').astype(str)
        label = 'FAKE' if '_fake_' in name.lower() else 'REAL'
        subject = name.replace('_news_content.csv', '').replace('.csv', '')
        frames.append(pd.DataFrame({
            'title': df['title'],
            'text': df['text'],
            'subject': subject,
            'date': df.get('publish_date', '').fillna('').astype(str),
            'label': label,
            'dataset_source': 'fakenewsnet',
            'source_group': subject,
        }))
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=8)
def _load_news_category_cached(base_dir_str: str) -> pd.DataFrame:
    df = pd.read_json(
        Path(base_dir_str) / 'data' / 'raw' / 'news_category' / 'News_Category_Dataset_v3.json',
        lines=True,
    )
    df['headline'] = df['headline'].fillna('').astype(str)
    df['short_description'] = df['short_description'].fillna('').astype(str)
    df['category'] = df['category'].fillna('Unknown').astype(str)
    df['title'] = df['headline']
    df['text'] = (df['headline'] + ' ' + df['short_description']).str.strip()
    df['date'] = df['date'].astype(str)
    df['subject'] = df['category']
    df['label'] = 'REAL'
    df['dataset_source'] = 'news_category_real_boost'
    df['source_group'] = df['category']
    return df[['title', 'text', 'subject', 'date', 'label', 'dataset_source', 'source_group']]


# ---------------------------------------------------------------------------
# Public API — return copies so callers can freely modify the DataFrame.
# ---------------------------------------------------------------------------

def load_kaggle_fake_real(base_dir: Path) -> pd.DataFrame:
    return _load_kaggle_cached(str(base_dir)).copy()


def load_fakenewsnet(base_dir: Path) -> pd.DataFrame:
    return _load_fakenewsnet_cached(str(base_dir)).copy()


def load_news_category(base_dir: Path) -> pd.DataFrame:
    return _load_news_category_cached(str(base_dir)).copy()


def build_classification_dataset(base_dir: Path, extra_real_rows: int | None = 4000) -> pd.DataFrame:
    """Build the classifier table from raw sources.

    The fake/real backbone always comes from the labeled raw datasets
    (Kaggle Fake/True + FakeNewsNet). ``extra_real_rows`` optionally adds a
    stratified sample from the News Category dataset as an additional REAL-only
    domain broadening source. Use ``0`` to disable the extra REAL boost and
    ``None`` to include all available News Category rows.
    """
    kaggle = load_kaggle_fake_real(base_dir)
    fnn = load_fakenewsnet(base_dir)
    frames = [kaggle, fnn]
    if extra_real_rows is None or extra_real_rows > 0:
        news = load_news_category(base_dir)
        news_extra = stratified_sample(news, 'subject', extra_real_rows, random_state=42)
        frames.append(news_extra)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df['combined_text'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).str.strip()
    df = df[df['combined_text'].str.len() > 30].copy()
    df['word_count'] = df['combined_text'].str.split().str.len().fillna(0).astype(int)
    df = df.drop_duplicates(subset=['combined_text']).reset_index(drop=True)
    return df


def build_topic_dataset(base_dir: Path, topic_rows: int | None = 10000) -> pd.DataFrame:
    # .copy() ensures we don't mutate the cached raw DataFrame
    topic_df = load_news_category(base_dir).copy()
    topic_df['combined_text'] = (topic_df['title'] + ' ' + topic_df['text']).str.strip()
    topic_df = topic_df[topic_df['combined_text'].str.len() > 30].copy()
    topic_df = stratified_sample(topic_df, 'subject', topic_rows, random_state=42)
    extra_path = base_dir / 'data' / 'topic_modeling_extra.csv'
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        extra['title'] = extra.get('title', '').fillna('').astype(str)
        extra['text'] = extra.get('text', '').fillna('').astype(str)
        extra['subject'] = extra.get('category', 'Extra').fillna('Extra').astype(str)
        extra['date'] = ''
        extra['label'] = 'UNLABELED'
        extra['dataset_source'] = 'topic_modeling_extra'
        extra['source_group'] = extra['subject']
        extra['combined_text'] = (extra['title'] + ' ' + extra['text']).str.strip()
        topic_df = pd.concat(
            [topic_df, extra[['title', 'text', 'subject', 'date', 'label', 'dataset_source', 'source_group', 'combined_text']]],
            ignore_index=True,
        )
    return topic_df.drop_duplicates(subset=['combined_text']).reset_index(drop=True)
