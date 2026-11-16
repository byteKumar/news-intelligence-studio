"""Model-loading and prediction helpers for fake/real classification."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd

_DEFAULT_BASE = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def _load_model_cached(base_dir: Path):
    """Load and cache the classifier pipeline from disk."""
    return joblib.load(base_dir / 'models' / 'fake_news_classifier.joblib')


def load_model(base_dir: Path | None = None):
    return _load_model_cached(Path(base_dir) if base_dir is not None else _DEFAULT_BASE)


def _term_contributions(model, text: str):
    vec = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    X = vec.transform([text])
    row = X.toarray()[0]
    active = np.where(row > 0)[0]
    names = vec.get_feature_names_out()
    if len(active) == 0:
        return pd.DataFrame(columns=['term', 'contribution'])
    if hasattr(clf, 'coef_'):
        coef = clf.coef_[0]
        contrib = row[active] * coef[active]
    elif hasattr(clf, 'feature_log_prob_') and len(clf.feature_log_prob_) >= 2:
        coef = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        contrib = row[active] * coef[active]
    else:
        return pd.DataFrame(columns=['term', 'contribution'])
    return pd.DataFrame({'term': names[active], 'contribution': contrib})


def explain_prediction(text: str, base_dir: Path | None = None, top_n: int = 10) -> dict:
    model = load_model(base_dir)
    df = _term_contributions(model, text)
    if df.empty:
        return {'fake_terms': [], 'real_terms': []}
    fake_terms = df.sort_values('contribution').head(top_n)
    real_terms = df.sort_values('contribution', ascending=False).head(top_n)
    fake_terms = fake_terms[fake_terms['contribution'] < 0].copy()
    real_terms = real_terms[real_terms['contribution'] > 0].copy()
    return {
        'fake_terms': fake_terms.round(4).to_dict(orient='records'),
        'real_terms': real_terms.round(4).to_dict(orient='records'),
    }


def predict_details(text: str, base_dir: Path | None = None) -> dict:
    model = load_model(base_dir)
    pred = model.predict([text])[0]
    classes = list(model.classes_)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba([text])[0]
    else:
        raw = model.decision_function([text])
        raw = float(raw[0] if hasattr(raw, '__len__') else raw)
        prob_real = 1.0 / (1.0 + math.exp(-raw))
        probs = np.array([1 - prob_real, prob_real]) if classes == ['FAKE', 'REAL'] else np.array([prob_real, 1 - prob_real])
    idx = classes.index(pred)
    margin = None
    if hasattr(model, 'decision_function'):
        raw = model.decision_function([text])
        margin = float(raw[0] if hasattr(raw, '__len__') else raw)
    return {
        'prediction': pred,
        'confidence': float(probs[idx]),
        'class_probabilities': [{'label': c, 'probability': float(p)} for c, p in zip(classes, probs)],
        'margin': margin,
        'explanation': explain_prediction(text, base_dir),
    }
