"""Helpers for loading topic artifacts and inferring the nearest theme."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import joblib
from .text_utils import normalize_text

_DEFAULT_BASE = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def _load_topic_artifacts_cached(base_dir: Path):
    """Load and cache the topic model artifacts from disk."""
    return joblib.load(base_dir / 'models' / 'topic_model_artifacts.joblib')


def load_topic_artifacts(base_dir: Path | None = None):
    return _load_topic_artifacts_cached(Path(base_dir) if base_dir is not None else _DEFAULT_BASE)


def infer_topic(text: str, base_dir: Path | None = None) -> dict:
    art = load_topic_artifacts(base_dir)
    vec = art['vectorizer']
    model = art.get('model', art.get('lda'))
    label_map = art['label_map']
    topic_keywords = {row['topic_id']: row['top_words'].split(', ') for row in art['topic_keywords']}
    clean = normalize_text(text)
    X = vec.transform([clean])
    model_type = art.get('model_type', 'lda')
    if model_type in {'lda', 'nmf'}:
        dist = model.transform(X)[0]
    elif model_type == 'kmeans':
        distances = model.transform(X)[0]
        sim = 1.0 / (distances + 1e-9)
        dist = sim / sim.sum()
    else:
        raise ValueError(f'Unsupported topic model type: {model_type}')
    topic_id = int(dist.argmax())
    ranked = sorted(
        [{'topic_id': int(i), 'topic_name': label_map.get(int(i), f'Topic {i}'), 'probability': float(v)}
         for i, v in enumerate(dist)],
        key=lambda r: r['probability'],
        reverse=True,
    )
    matched = [w for w in topic_keywords.get(topic_id, []) if w in clean.split()]
    confidence_gap = (
        float(ranked[0]['probability'] - ranked[1]['probability'])
        if len(ranked) > 1
        else float(ranked[0]['probability'])
    )
    return {
        'topic_id': topic_id,
        'topic_name': label_map.get(topic_id, f'Topic {topic_id}'),
        'score': float(dist[topic_id]),
        'matched_words': matched,
        'top_topics': ranked[:5],
        'confidence_gap': confidence_gap,
        'model_type': model_type,
    }
