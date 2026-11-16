"""Training, benchmarking, and asset-building code for the project."""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data_loader import build_classification_dataset, build_topic_dataset, balanced_label_sample, load_kaggle_fake_real, load_fakenewsnet, load_news_category
from .pipeline import predict_article
from .summarizer import summarize
from .text_utils import normalize_text

Callback = Callable[[str, dict], None]
TOPIC_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    'said','would','also','could','one','two','new','year','years','time','first','last','say','says',
    'news','video','photos','just','really','need','want','like','today','people','according','u','s','us',
    'reuters','mr','ms','told','make','made','get','got','go','going','take','took','still','amp','ll','ve',
    'don','didn','doesn','won','twitter','facebook','instagram'
}


def _emit(callback: Callback | None, stage: str, payload: dict):
    """Send progress/status events to optional UI callback."""
    if callback is not None:
        callback(stage, payload)


def ensure_dirs(base_dir: Path):
    """Create required output/input directories if they do not exist."""
    for sub in ['results','results/figures','models','data/processed','data/raw/kaggle_fake_real','data/raw/fakenewsnet','data/raw/news_category']:
        (base_dir / sub).mkdir(parents=True, exist_ok=True)


def _metric_bundle(y_true, preds) -> dict:
    """Return standard classification metrics used for model comparison."""
    acc = accuracy_score(y_true, preds)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, preds, average='weighted', zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, preds, average='macro', zero_division=0)
    return {
        'accuracy': float(acc),
        'precision_weighted': float(p_w), 'recall_weighted': float(r_w), 'f1_weighted': float(f_w),
        'precision_macro': float(p_m), 'recall_macro': float(r_m), 'f1_macro': float(f_m)
    }


def _models():
    """Define candidate classifier pipelines for benchmark selection."""
    vectorizer = dict(stop_words='english', max_features=9000, min_df=4, ngram_range=(1,2), sublinear_tf=True)
    return {
        'logistic_regression': Pipeline([
            ('tfidf', TfidfVectorizer(**vectorizer)),
            ('clf', LogisticRegression(max_iter=1200, C=4.0, class_weight='balanced', solver='liblinear', random_state=42))
        ]),
        'linear_svc': Pipeline([
            ('tfidf', TfidfVectorizer(**vectorizer)),
            ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=42))
        ]),
        'multinomial_nb': Pipeline([
            ('tfidf', TfidfVectorizer(**vectorizer)),
            ('clf', MultinomialNB(alpha=0.12))
        ]),
        'sgd_log_loss': Pipeline([
            ('tfidf', TfidfVectorizer(**vectorizer)),
            ('clf', SGDClassifier(loss='log_loss', alpha=1e-5, penalty='l2', class_weight='balanced', random_state=42))
        ]),
    }


def save_overview(class_df: pd.DataFrame, topic_df: pd.DataFrame, base_dir: Path, dataset_profile: dict | None = None):
    """Write dataset overview JSON and preview CSV artifacts."""
    overview = {
        'classification_rows': int(len(class_df)),
        'topic_rows': int(len(topic_df)),
        'classification_source_breakdown': {k: int(v) for k, v in class_df['dataset_source'].value_counts().to_dict().items()},
        'classification_label_balance': {k: int(v) for k, v in class_df['label'].value_counts().to_dict().items()},
        'topic_category_top20': {k: int(v) for k, v in topic_df['subject'].value_counts().head(20).to_dict().items()},
        'column_explanations': {
            'title': 'Headline of the article.',
            'text': 'Article body or best available descriptive text.',
            'subject': 'Original subject/category from the source dataset.',
            'date': 'Original date string if available.',
            'label': 'Target label, FAKE or REAL.',
            'dataset_source': 'Which raw dataset contributed the row.',
            'source_group': 'A finer source or category grouping.',
            'combined_text': 'Title and text joined for modeling.',
            'word_count': 'Approximate token count of combined_text.'
        },
        'raw_data_locations': {
            'kaggle_fake_real': 'data/raw/kaggle_fake_real/Fake.csv and True.csv',
            'fakenewsnet': 'data/raw/fakenewsnet/*.csv',
            'news_category': 'data/raw/news_category/News_Category_Dataset_v3.json'
        },
        'selection_policy': {
            'classifier': 'Choose the model with the best validation macro F1, then weighted F1, then accuracy. Speed is shown but only used as a late tie-breaker.',
            'topic_model': 'Choose the topic model with the highest quality score based on coherence and topic diversity. Training time is shown separately but is not the main criterion.'
        }
    }
    if dataset_profile is not None:
        overview['dataset_profile'] = dataset_profile
    (base_dir / 'results' / 'dataset_overview.json').write_text(json.dumps(overview, indent=2), encoding='utf-8')
    class_df.head(1200).to_csv(base_dir / 'data' / 'processed' / 'classification_preview.csv', index=False)
    topic_df.head(1200).to_csv(base_dir / 'data' / 'processed' / 'topic_preview.csv', index=False)
    class_df['dataset_source'].value_counts().rename_axis('dataset_source').reset_index(name='rows').to_csv(base_dir / 'results' / 'source_breakdown.csv', index=False)
    topic_df['subject'].value_counts().head(30).rename_axis('subject').reset_index(name='rows').to_csv(base_dir / 'results' / 'topic_input_category_counts.csv', index=False)


def _select_best_classifier(benchmark_df: pd.DataFrame) -> pd.Series:
    """Select top classifier by macro F1, then weighted F1, then accuracy."""
    ranked = benchmark_df.sort_values(['f1_macro', 'f1_weighted', 'accuracy', 'train_time_sec'], ascending=[False, False, False, True]).reset_index(drop=True)
    return ranked.iloc[0]


def train_classifier(class_df: pd.DataFrame, base_dir: Path, callback: Callback | None = None):
    """Benchmark classifier candidates, persist best model, and save reports.

    Args:
        class_df: Prepared classification dataset with `combined_text` and labels.
        base_dir: Project root for writing model/results artifacts.
        callback: Optional progress callback used by Streamlit training UI.

    Returns:
        Summary dict with selected model and held-out test metrics.
    """
    X = class_df['combined_text']
    y = class_df['label']
    meta = class_df[['title', 'dataset_source', 'subject']].reset_index(drop=True)
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(X, y, meta, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(X_temp, y_temp, meta_temp, test_size=0.50, stratify=y_temp, random_state=42)

    split_table = pd.DataFrame([
        {'split': 'train', 'rows': int(len(X_train)), 'fake_rows': int((y_train == 'FAKE').sum()), 'real_rows': int((y_train == 'REAL').sum())},
        {'split': 'validation', 'rows': int(len(X_val)), 'fake_rows': int((y_val == 'FAKE').sum()), 'real_rows': int((y_val == 'REAL').sum())},
        {'split': 'test', 'rows': int(len(X_test)), 'fake_rows': int((y_test == 'FAKE').sum()), 'real_rows': int((y_test == 'REAL').sum())},
    ])
    split_table.to_csv(base_dir / 'results' / 'classification_split_table.csv', index=False)

    rows = []
    fitted = {}
    models = _models()
    total = len(models)
    global_start = time.perf_counter()
    for idx, (name, pipe) in enumerate(models.items(), start=1):
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - start
        pred_start = time.perf_counter()
        val_preds = pipe.predict(X_val)
        predict_time = time.perf_counter() - pred_start
        metrics = _metric_bundle(y_val, val_preds)
        rows.append({
            'model_name': name,
            **metrics,
            'train_time_sec': round(float(train_time), 4),
            'predict_time_ms_per_article': round(float((predict_time / max(len(X_val), 1)) * 1000), 4),
        })
        fitted[name] = pipe
        elapsed = time.perf_counter() - global_start
        eta = (elapsed / idx) * (total - idx) if idx < total else 0.0
        _emit(callback, 'classifier_progress', {
            'current_model': name,
            'completed_models': idx,
            'total_models': total,
            'elapsed_sec': float(elapsed),
            'eta_sec': float(eta),
            'benchmark_df': pd.DataFrame(rows).sort_values('f1_macro', ascending=False)
        })

    benchmark_df = pd.DataFrame(rows)
    best_row = _select_best_classifier(benchmark_df)
    best_name = str(best_row['model_name'])
    best_pipe = fitted[best_name]
    best_pipe.fit(pd.concat([X_train, X_val], ignore_index=True), pd.concat([y_train, y_val], ignore_index=True))
    test_preds = best_pipe.predict(X_test)
    metrics = _metric_bundle(y_test, test_preds)

    benchmark_df = benchmark_df.sort_values(['f1_macro', 'f1_weighted', 'accuracy'], ascending=False)
    benchmark_df.to_csv(base_dir / 'results' / 'model_benchmark.csv', index=False)
    selection = {
        'selected_model': best_name,
        'selection_reason': 'Highest validation macro F1, with weighted F1 and accuracy as secondary checks. Speed is shown for context but not used as the main criterion.',
        'selected_row': json.loads(pd.DataFrame([best_row]).to_json(orient='records'))[0]
    }
    (base_dir / 'results' / 'classifier_selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')

    cm = confusion_matrix(y_test, test_preds, labels=['FAKE','REAL'])
    pd.DataFrame(cm, index=['actual_FAKE','actual_REAL'], columns=['pred_FAKE','pred_REAL']).to_csv(base_dir / 'results' / 'confusion_matrix.csv')
    plt.figure(figsize=(5,4))
    plt.imshow(cm)
    plt.xticks([0,1], ['FAKE','REAL'])
    plt.yticks([0,1], ['FAKE','REAL'])
    plt.title(f'Confusion Matrix ({best_name})')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i,j]), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(base_dir / 'results' / 'figures' / 'confusion_matrix.png', dpi=200)
    plt.close()

    report = classification_report(y_test, test_preds, output_dict=True)
    (base_dir / 'results' / 'classification_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    summary = {
        'best_model_name': best_name,
        'test_metrics': metrics,
        'splits': {'train_rows': int(len(X_train)), 'validation_rows': int(len(X_val)), 'test_rows': int(len(X_test))},
        'selection_reason': selection['selection_reason']
    }
    (base_dir / 'results' / 'classification_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    sample_predictions = pd.DataFrame({
        'title': class_df.loc[X_test.index, 'title'].reset_index(drop=True),
        'actual': y_test.reset_index(drop=True),
        'predicted': pd.Series(test_preds),
        'dataset_source': meta_test['dataset_source'].reset_index(drop=True),
        'subject': meta_test['subject'].reset_index(drop=True)
    })
    sample_predictions.to_csv(base_dir / 'results' / 'sample_test_predictions.csv', index=False)
    perf = []
    for source, sub in sample_predictions.groupby('dataset_source'):
        perf.append({'dataset_source': source, 'rows': int(len(sub)), **_metric_bundle(sub['actual'], sub['predicted'])})
    perf_df = pd.DataFrame(perf).sort_values('rows', ascending=False)
    perf_df.to_csv(base_dir / 'results' / 'performance_by_source.csv', index=False)
    joblib.dump(best_pipe, base_dir / 'models' / 'fake_news_classifier.joblib')
    _emit(callback, 'classifier_done', {
        'benchmark_df': benchmark_df,
        'performance_by_source': perf_df,
        'test_metrics': metrics,
        'best_model_name': best_name,
        'selection': selection,
    })
    return summary


def _pretty(cat: str) -> str:
    """Normalize category text into title case for display labels."""
    return ' '.join(str(cat).replace('_', ' ').split()).title()


def _top_words_from_components(components: np.ndarray, feature_names: np.ndarray, top_n: int = 10) -> list[list[str]]:
    """Extract top weighted words per topic/component."""
    out = []
    for comp in components:
        top_idx = np.argsort(comp)[-top_n:][::-1]
        out.append([feature_names[i] for i in top_idx])
    return out


def _binary_doc_presence(X: sparse.spmatrix) -> sparse.csr_matrix:
    """Convert sparse term matrix into binary document-term presence matrix."""
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    X = X.copy()
    X.data = np.ones_like(X.data)
    return X


def _coherence_proxy(binary_X: sparse.csr_matrix, top_word_lists: list[list[str]], vocab_lookup: dict[str, int]) -> float:
    """Compute a lightweight coherence proxy from term co-occurrence."""
    doc_freq = np.asarray(binary_X.sum(axis=0)).ravel()
    scores = []
    for words in top_word_lists:
        idxs = [vocab_lookup[w] for w in words if w in vocab_lookup]
        for i in range(1, len(idxs)):
            for j in range(i):
                wi = idxs[i]
                wj = idxs[j]
                cooc = binary_X[:, wi].multiply(binary_X[:, wj]).sum()
                denom = max(float(doc_freq[wj]), 1.0)
                scores.append(math.log((float(cooc) + 1.0) / denom))
    return float(np.mean(scores)) if scores else -99.0


def _topic_diversity(top_word_lists: list[list[str]]) -> float:
    """Compute diversity as unique top words over total top-word slots."""
    flat = [w for words in top_word_lists for w in words]
    return float(len(set(flat)) / max(len(flat), 1))


def _fit_topic_candidate(model_name: str, clean_text: pd.Series, n_topics: int):
    """Train one topic-model candidate and return quality diagnostics."""
    if model_name == 'lda':
        vectorizer = CountVectorizer(stop_words=list(TOPIC_STOPWORDS), max_features=3200, min_df=8, max_df=0.85, ngram_range=(1, 2))
        X = vectorizer.fit_transform(clean_text)
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch', max_iter=8, doc_topic_prior=0.6, topic_word_prior=0.08)
        doc_topic = model.fit_transform(X)
        components = model.components_
    elif model_name == 'nmf':
        vectorizer = TfidfVectorizer(stop_words=list(TOPIC_STOPWORDS), max_features=4200, min_df=6, max_df=0.85, ngram_range=(1, 2), sublinear_tf=True)
        X = vectorizer.fit_transform(clean_text)
        model = NMF(n_components=n_topics, init='nndsvda', random_state=42, max_iter=350)
        doc_topic = model.fit_transform(X)
        components = model.components_
    elif model_name == 'kmeans':
        vectorizer = TfidfVectorizer(stop_words=list(TOPIC_STOPWORDS), max_features=4200, min_df=6, max_df=0.85, ngram_range=(1, 2), sublinear_tf=True)
        X = vectorizer.fit_transform(clean_text)
        model = KMeans(n_clusters=n_topics, n_init=10, random_state=42)
        model.fit(X)
        distances = model.transform(X)
        similarities = 1.0 / (distances + 1e-9)
        doc_topic = similarities / similarities.sum(axis=1, keepdims=True)
        components = model.cluster_centers_
    else:
        raise ValueError(model_name)

    feature_names = vectorizer.get_feature_names_out()
    top_word_lists = _top_words_from_components(components, feature_names, top_n=10)
    binary_X = _binary_doc_presence(X)
    vocab_lookup = {term: i for i, term in enumerate(feature_names)}
    coherence = _coherence_proxy(binary_X, top_word_lists, vocab_lookup)
    diversity = _topic_diversity(top_word_lists)
    avg_conf = float(np.mean(doc_topic.max(axis=1)))
    return {
        'model_name': model_name,
        'vectorizer': vectorizer,
        'model': model,
        'doc_topic': doc_topic,
        'top_word_lists': top_word_lists,
        'coherence_proxy': coherence,
        'topic_diversity': diversity,
        'avg_topic_confidence': avg_conf,
    }


def _normalize_series(values: pd.Series) -> pd.Series:
    """Min-max normalize a metric series for weighted quality scoring."""
    if values.nunique() <= 1:
        return pd.Series([1.0] * len(values), index=values.index)
    return (values - values.min()) / (values.max() - values.min())


def _label_topics(topic_df: pd.DataFrame, doc_topic: np.ndarray, top_word_lists: list[list[str]]) -> tuple[dict[int, str], list[dict], pd.DataFrame]:
    """Build readable topic labels, keyword rows, and representative-title rows."""
    topic_assign = doc_topic.argmax(axis=1)
    label_map: dict[int, str] = {}
    keyword_rows: list[dict] = []
    rep_rows = []
    for topic_idx, words in enumerate(top_word_lists):
        best_docs = np.argsort(doc_topic[:, topic_idx])[-6:][::-1]
        reps = topic_df.iloc[best_docs]
        top_cats = [_pretty(v) for v in reps['subject'].value_counts().head(2).index.tolist()]
        label_root = ' / '.join(top_cats) if top_cats else f'Topic {topic_idx}'
        label = f"{label_root} — {', '.join(words[:4])}"
        label_map[int(topic_idx)] = label
        keyword_rows.append({
            'topic_id': int(topic_idx),
            'topic_name': label,
            'top_words': ', '.join(words),
            'top_categories': ' | '.join(top_cats),
            'representative_titles': ' | '.join(reps['title'].head(3).tolist())
        })
        for rank, (_, row) in enumerate(reps.iterrows(), start=1):
            rep_rows.append({
                'topic_id': int(topic_idx),
                'topic_name': label,
                'rank_within_topic': rank,
                'subject': row['subject'],
                'title': row['title'],
                'topic_score': float(doc_topic[row.name, topic_idx])
            })
    return label_map, keyword_rows, pd.DataFrame(rep_rows)


def train_topic_models(topic_df: pd.DataFrame, base_dir: Path, n_topics: int = 12, callback: Callback | None = None):
    """Benchmark topic candidates, select best, and persist topic artifacts.

    Selection uses weighted quality score:
    0.60 * coherence + 0.25 * diversity + 0.15 * avg topic confidence.
    """
    topic_df = topic_df.copy().reset_index(drop=True)
    topic_df['clean_text'] = topic_df['combined_text'].map(normalize_text)
    candidate_names = ['lda', 'nmf', 'kmeans']
    results = []
    fitted = {}
    global_start = time.perf_counter()
    for idx, name in enumerate(candidate_names, start=1):
        start = time.perf_counter()
        fit = _fit_topic_candidate(name, topic_df['clean_text'], n_topics=n_topics)
        fit_time = time.perf_counter() - start
        results.append({
            'model_name': name,
            'coherence_proxy': fit['coherence_proxy'],
            'topic_diversity': fit['topic_diversity'],
            'avg_topic_confidence': fit['avg_topic_confidence'],
            'train_time_sec': round(float(fit_time), 4),
        })
        fitted[name] = fit
        elapsed = time.perf_counter() - global_start
        eta = (elapsed / idx) * (len(candidate_names) - idx) if idx < len(candidate_names) else 0.0
        _emit(callback, 'topic_progress', {
            'current_model': name,
            'completed_models': idx,
            'total_models': len(candidate_names),
            'elapsed_sec': float(elapsed),
            'eta_sec': float(eta),
            'benchmark_df': pd.DataFrame(results)
        })

    bench = pd.DataFrame(results)
    bench['coherence_norm'] = _normalize_series(bench['coherence_proxy'])
    bench['diversity_norm'] = _normalize_series(bench['topic_diversity'])
    bench['confidence_norm'] = _normalize_series(bench['avg_topic_confidence'])
    bench['quality_score'] = (0.60 * bench['coherence_norm'] + 0.25 * bench['diversity_norm'] + 0.15 * bench['confidence_norm']).round(4)
    bench = bench.sort_values(['quality_score', 'coherence_proxy', 'topic_diversity'], ascending=False).reset_index(drop=True)
    bench.to_csv(base_dir / 'results' / 'topic_model_benchmark.csv', index=False)
    best_name = str(bench.iloc[0]['model_name'])
    best = fitted[best_name]

    label_map, keyword_rows, rep_df = _label_topics(topic_df, best['doc_topic'], best['top_word_lists'])
    keywords = pd.DataFrame(keyword_rows)
    rep_df.to_csv(base_dir / 'results' / 'topic_representative_titles.csv', index=False)
    keywords.to_csv(base_dir / 'results' / 'topic_keywords.csv', index=False)
    topic_assignments = topic_df.copy()
    topic_assignments['topic_id'] = best['doc_topic'].argmax(axis=1)
    topic_assignments['topic_name'] = topic_assignments['topic_id'].map(label_map)
    topic_assignments.head(1000).to_csv(base_dir / 'results' / 'topic_assignments_sample.csv', index=False)
    dist = topic_assignments['topic_name'].value_counts().rename_axis('topic_name').reset_index(name='rows')
    dist.to_csv(base_dir / 'results' / 'topic_distribution.csv', index=False)
    plt.figure(figsize=(9, 6))
    dist.sort_values('rows').plot(kind='barh', x='topic_name', y='rows', legend=False, title=f'Topic distribution ({best_name})')
    plt.tight_layout()
    plt.savefig(base_dir / 'results' / 'figures' / 'topic_distribution.png', dpi=200)
    plt.close()

    topic_selection = {
        'selected_model': best_name,
        'selection_reason': 'Chosen by the highest topic-model quality score, which prioritizes coherence and topic diversity over runtime.',
        'n_topics': int(n_topics),
        'selected_row': json.loads(pd.DataFrame([bench.iloc[0]]).to_json(orient='records'))[0]
    }
    (base_dir / 'results' / 'topic_selection.json').write_text(json.dumps(topic_selection, indent=2), encoding='utf-8')

    joblib.dump({
        'model_type': best_name,
        'vectorizer': best['vectorizer'],
        'model': best['model'],
        'label_map': label_map,
        'topic_keywords': keyword_rows,
        'n_topics': n_topics,
    }, base_dir / 'models' / 'topic_model_artifacts.joblib')
    _emit(callback, 'topic_done', {
        'topic_distribution': dist,
        'topic_keywords': keywords,
        'benchmark_df': bench,
        'selection': topic_selection,
    })
    return {'benchmark': bench, 'selection': topic_selection}


def build_summary_examples(class_df: pd.DataFrame, base_dir: Path):
    """Generate sample extractive summaries and compression statistics."""
    sample_pool = class_df[class_df['text'].str.split().str.len() > 120]
    if sample_pool.empty:
        sample_pool = class_df.copy()
    sample = sample_pool.sample(min(10, len(sample_pool)), random_state=42)[['title', 'text', 'label', 'dataset_source', 'subject']].copy()
    sample['generated_summary'] = sample['text'].map(lambda x: summarize(x, 3))
    sample['original_word_count'] = sample['text'].str.split().str.len()
    sample['summary_word_count'] = sample['generated_summary'].str.split().str.len()
    sample['compression_ratio'] = (sample['summary_word_count'] / sample['original_word_count']).round(3)
    sample.to_csv(base_dir / 'results' / 'sample_summaries.csv', index=False)
    return sample


def build_demo_outputs(class_df: pd.DataFrame, base_dir: Path):
    """Run small batch inference and save pipeline demo output table."""
    rows = []
    for _, row in class_df.sample(min(8, len(class_df)), random_state=7)[['title','text','label','dataset_source']].iterrows():
        out = predict_article(row['title'], row['text'], base_dir=base_dir)
        rows.append({'actual_label': row['label'], 'predicted_label': out['prediction'], 'confidence': out['confidence'], 'topic_name': out['topic']['topic_name'], 'topic_score': out['topic']['score'], 'dataset_source': row['dataset_source'], 'summary': out['summary'], 'title': row['title']})
    pd.DataFrame(rows).to_csv(base_dir / 'results' / 'pipeline_demo_outputs.csv', index=False)


def _dataset_profile_from_raw(base_dir: Path) -> dict:
    """Collect raw-source statistics for reporting tables and appendices."""
    kaggle = load_kaggle_fake_real(base_dir)
    fakenewsnet = load_fakenewsnet(base_dir)
    news_category = load_news_category(base_dir)
    return {
        'kaggle_fake_real': {
            'rows': int(len(kaggle)),
            'fake_rows': int((kaggle['label'] == 'FAKE').sum()),
            'real_rows': int((kaggle['label'] == 'REAL').sum()),
            'subjects': int(kaggle['subject'].nunique()),
        },
        'fakenewsnet': {
            'rows': int(len(fakenewsnet)),
            'fake_rows': int((fakenewsnet['label'] == 'FAKE').sum()),
            'real_rows': int((fakenewsnet['label'] == 'REAL').sum()),
            'source_groups': int(fakenewsnet['source_group'].nunique()),
        },
        'news_category': {
            'rows': int(len(news_category)),
            'categories': int(news_category['subject'].nunique()),
        },
    }


def build_project_assets(
    base_dir: Path,
    classifier_rows: int | None = None,
    extra_real_rows: int | None = 4000,
    topic_rows: int | None = None,
    n_topics: int = 12,
    callback: Callback | None = None,
):
    """Rebuild end-to-end project artifacts from raw data.

    Pipeline steps:
    1) build/sampling classification + topic tables
    2) benchmark/select classifier
    3) benchmark/select topic model
    4) save reports, figures, model artifacts, and demo outputs

    Returns:
        Build summary dict used by dashboard and header metrics.
    """
    ensure_dirs(base_dir)
    _emit(callback, 'status', {'message': 'Loading raw datasets and building training tables...'})
    all_class = build_classification_dataset(base_dir, extra_real_rows=extra_real_rows)
    train_class = balanced_label_sample(all_class, classifier_rows, random_state=42)
    topic_df = build_topic_dataset(base_dir, topic_rows=topic_rows)
    dataset_profile = _dataset_profile_from_raw(base_dir)
    save_overview(train_class, topic_df, base_dir, dataset_profile=dataset_profile)
    _emit(callback, 'data_loaded', {
        'class_source_counts': train_class['dataset_source'].value_counts().to_dict(),
        'topic_category_counts': topic_df['subject'].value_counts().head(20).to_dict(),
    })
    _emit(callback, 'status', {'message': 'Benchmarking classifier models...'})
    cls = train_classifier(train_class, base_dir, callback=callback)
    _emit(callback, 'status', {'message': 'Benchmarking topic models...'})
    topic_info = train_topic_models(topic_df, base_dir, n_topics=n_topics, callback=callback)
    _emit(callback, 'status', {'message': 'Generating summaries and demo predictions...'})
    build_summary_examples(train_class[train_class['dataset_source'] != 'news_category_real_boost'], base_dir)
    build_demo_outputs(train_class, base_dir)
    summary = {
        'classifier_rows_used': int(len(train_class)),
        'extra_real_rows_requested': None if extra_real_rows is None else int(extra_real_rows),
        'topic_rows_used': int(len(topic_df)),
        'n_topics': int(n_topics),
        'best_model': cls['best_model_name'],
        'best_topic_model': topic_info['selection']['selected_model'],
        'test_f1_weighted': float(cls['test_metrics']['f1_weighted']),
        'test_f1_macro': float(cls['test_metrics']['f1_macro']),
        'classifier_selection_reason': cls['selection_reason'],
        'topic_selection_reason': topic_info['selection']['selection_reason'],
        'raw_data_profile': dataset_profile,
    }
    (base_dir / 'results' / 'build_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _emit(callback, 'build_done', {'summary': summary})
    return summary


def main():
    """CLI entry point for one-shot project rebuild."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--classifier-rows', type=int, default=None)
    parser.add_argument('--extra-real-rows', type=int, default=4000)
    parser.add_argument('--topic-rows', type=int, default=None)
    parser.add_argument('--n-topics', type=int, default=12)
    args = parser.parse_args()
    out = build_project_assets(Path(args.base_dir).resolve(), classifier_rows=args.classifier_rows, extra_real_rows=args.extra_real_rows, topic_rows=args.topic_rows, n_topics=args.n_topics)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
