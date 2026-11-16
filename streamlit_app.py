from __future__ import annotations
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import predict_article
from src.train_pipeline import build_project_assets
from fact_verify.pipeline import FactVerificationPipeline
from fact_verify.loaders import build_sample_articles_from_raw

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Fake News Detector', layout='wide')


# Helper functions

def read_json(path: Path):
    """Return parsed JSON, or None when file is missing."""
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

def read_csv(path: Path):
    """Return CSV as DataFrame, or empty DataFrame when missing."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def artifacts_ready():
    """Check whether trained artifacts required by Analyze tab exist."""
    return all((BASE / p).exists() for p in [
        'models/fake_news_classifier.joblib',
        'models/topic_model_artifacts.joblib',
        'results/classification_metrics.json',
    ])

def raw_data_status():
    """Return availability map for required raw dataset files."""
    required = {
        'kaggle_fake_real/Fake.csv': BASE / 'data/raw/kaggle_fake_real/Fake.csv',
        'kaggle_fake_real/True.csv': BASE / 'data/raw/kaggle_fake_real/True.csv',
        'fakenewsnet/BuzzFeed_fake_news_content.csv': BASE / 'data/raw/fakenewsnet/BuzzFeed_fake_news_content.csv',
        'fakenewsnet/BuzzFeed_real_news_content.csv': BASE / 'data/raw/fakenewsnet/BuzzFeed_real_news_content.csv',
        'fakenewsnet/PolitiFact_fake_news_content.csv': BASE / 'data/raw/fakenewsnet/PolitiFact_fake_news_content.csv',
        'fakenewsnet/PolitiFact_real_news_content.csv': BASE / 'data/raw/fakenewsnet/PolitiFact_real_news_content.csv',
        'news_category/News_Category_Dataset_v3.json': BASE / 'data/raw/news_category/News_Category_Dataset_v3.json',
    }
    return {name: path.exists() for name, path in required.items()}

def fmt_seconds(sec):
    """Pretty-print elapsed seconds for progress messages."""
    if sec is None:
        return '—'
    if sec < 60:
        return f'{sec:.1f}s'
    mins = int(sec // 60)
    return f'{mins}m {sec - mins * 60:.0f}s'

def confidence_label(score):
    """Map numeric confidence to UI bands: High/Moderate/Low."""
    if score >= 0.85:
        return 'High'
    if score >= 0.65:
        return 'Moderate'
    return 'Low'

def topic_confidence_label(score, gap):
    """Map topic score and score-gap to user-facing confidence level."""
    if score >= 0.42 and gap >= 0.12:
        return 'High'
    if score >= 0.28 and gap >= 0.08:
        return 'Moderate'
    return 'Low'

def simplify_topic_label(topic_name, matched_words=None):
    """Map raw topic labels to simpler readable names."""
    raw = str(topic_name or '').split('—')[0].strip()
    parts = [p.strip() for p in re.split(r'[/,&]+', raw) if p.strip()]
    mapping = {
        'politics': 'Public affairs', 'politics news': 'Public affairs',
        'worldpost': 'World affairs', 'world news': 'World affairs',
        'u.s. news': 'Public affairs',
        'black voices': 'Society & identity', 'latino voices': 'Society & identity',
        'queer voices': 'Society & identity',
        'style beauty': 'Lifestyle', 'style': 'Lifestyle', 'beauty': 'Lifestyle',
        'healthy living': 'Health & wellness', 'wellness': 'Health & wellness',
        'weddings': 'Lifestyle', 'parenting': 'Family & society',
        'parents': 'Family & society', 'women': 'Society & identity',
        'crime': 'Public safety', 'courts': 'Public safety',
        'business': 'Business', 'money': 'Business',
        'tech': 'Technology', 'science': 'Science',
        'environment': 'Climate & environment', 'green': 'Climate & environment',
        'travel': 'Travel', 'food': 'Lifestyle', 'taste': 'Lifestyle',
        'arts': 'Culture', 'culture arts': 'Culture',
        'entertainment': 'Entertainment', 'media': 'Media',
        'comedy': 'Entertainment', 'sports': 'Sports',
        'education': 'Education', 'religion': 'Society & belief',
    }
    broad = []
    for part in parts:
        key = part.lower().replace('_', ' ').strip()
        label = mapping.get(key)
        if label is None:
            if any(w in key for w in ['court', 'police', 'crime']):
                label = 'Public safety'
            elif any(w in key for w in ['polit', 'president', 'obama', 'trump', 'senate', 'congress']):
                label = 'Public affairs'
            elif any(w in key for w in ['health', 'wellness']):
                label = 'Health & wellness'
            elif any(w in key for w in ['entertain', 'celeb', 'music', 'movie']):
                label = 'Entertainment'
            else:
                label = part.title()
        if label not in broad:
            broad.append(label)
    if not broad and matched_words:
        words = set(matched_words or [])
        if words & {'iran', 'israel', 'missile', 'war', 'military', 'syria'}:
            broad.append('World affairs')
        elif words & {'court', 'president', 'senate', 'obama', 'trump', 'police'}:
            broad.append('Public affairs')
        elif words & {'market', 'company', 'tax', 'trade'}:
            broad.append('Business')
        elif words & {'movie', 'music', 'game', 'actor'}:
            broad.append('Entertainment')
    return ' / '.join(broad[:2]) if broad else 'General news'

def topic_rows_for_display(topic_dict):
    """Build compact top-topic table for chart rendering."""
    rows = []
    for row in topic_dict.get('top_topics', [])[:3]:
        rows.append({
            'Theme': simplify_topic_label(row.get('topic_name', ''), topic_dict.get('matched_words', [])),
            'Score': round(float(row.get('probability', 0.0)), 4),
            'Raw label': row.get('topic_name', ''),
        })
    return pd.DataFrame(rows)

def safe_scatter_fig(df, *, x, y, color=None, size=None, hover_data=None, title=''):
    """Safely create a scatter figure after numeric/column validation."""
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    plot_df = df.copy()
    for col in [x, y]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        plot_df = plot_df[plot_df[col].notna()]
    if plot_df.empty:
        return None
    kwargs = {'data_frame': plot_df, 'x': x, 'y': y, 'title': title}
    if color and color in plot_df.columns:
        kwargs['color'] = color
    valid_hover = [c for c in (hover_data or []) if c in plot_df.columns]
    if valid_hover:
        kwargs['hover_data'] = valid_hover
    if size and size in plot_df.columns:
        size_series = pd.to_numeric(plot_df[size], errors='coerce')
        valid_size = size_series.notna() & (size_series > 0)
        if valid_size.any():
            plot_df = plot_df.loc[valid_size].copy()
            plot_df[size] = size_series.loc[valid_size]
            kwargs['data_frame'] = plot_df
            kwargs['size'] = size
    if kwargs['data_frame'].empty:
        return None
    return px.scatter(**kwargs)

@st.cache_resource
def get_fact_verification_pipeline():
    """Create and cache the fact verification pipeline instance."""
    return FactVerificationPipeline(BASE)


# Page header

st.title('Fake News Detection & Analysis')
st.write('Classification, topic modeling, summarization, and fact verification in one pipeline.')

build_snapshot = read_json(BASE / 'results' / 'build_summary.json')

if build_snapshot:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Classifier rows', f"{build_snapshot.get('classifier_rows_used', '—'):,}")
    c2.metric('Best classifier', build_snapshot.get('best_model', '—'))
    c3.metric('Macro F1', f"{build_snapshot.get('test_f1_macro', 0):.4f}")
    c4.metric('Topic model', build_snapshot.get('best_topic_model', '—'))
else:
    st.info('No training results found yet. Use the Train tab to run a benchmark.')

# Sidebar

with st.sidebar:
    st.header('Training settings')
    classifier_rows = st.slider('Classifier training rows', 8000, 22000, 14000, 2000)
    extra_real_rows = st.slider('Extra REAL rows (HuffPost)', 0, 10000, 2000, 1000)
    topic_rows = st.slider('Topic modeling rows', 5000, 22000, 14000, 2500)
    n_topics = st.slider('Number of topics', 8, 18, 12, 1)
    st.divider()
    st.caption('Classifier picks the model with the best macro F1.')
    st.caption('Topic model picks the one with the best coherence + diversity score.')


# Tabs

train_tab, analyze_tab, verify_tab, dashboard_tab, data_tab = st.tabs([
    'Train', 'Analyze', 'Verify', 'Dashboard', 'Data'
])


# Train tab

with train_tab:
    st.subheader('Train & benchmark models')
    st.write('Compare several classifiers and topic models, then keep the best one based on quality metrics.')

    raw_status = raw_data_status()
    missing = [name for name, ok in raw_status.items() if not ok]

    if missing:
        st.warning(f'{len(missing)} required dataset file(s) are missing. Add them under `data/raw/` first.')
        with st.expander('Show missing files'):
            for f in missing:
                st.code(f)
    else:
        st.success('All dataset files found. Ready to train.')

    with st.expander('Required folder structure'):
        st.code("""data/raw/kaggle_fake_real/Fake.csv
data/raw/kaggle_fake_real/True.csv
data/raw/fakenewsnet/BuzzFeed_fake_news_content.csv
data/raw/fakenewsnet/BuzzFeed_real_news_content.csv
data/raw/fakenewsnet/PolitiFact_fake_news_content.csv
data/raw/fakenewsnet/PolitiFact_real_news_content.csv
data/raw/news_category/News_Category_Dataset_v3.json""")

    progress = st.progress(0)
    status_box = st.empty()
    timer_box = st.empty()
    chart_area = st.empty()
    summary_box = st.empty()

    run_clicked = st.button(
        'Start training', type='primary',
        width='stretch', disabled=bool(missing)
    )

    if run_clicked:
        def callback(stage, payload):
            if stage == 'status':
                status_box.info(payload['message'])
            elif stage == 'data_loaded':
                progress.progress(12)
            elif stage == 'classifier_progress':
                frac = 12 + int(35 * payload['completed_models'] / payload['total_models'])
                progress.progress(frac)
                timer_box.info(
                    f"Training classifier: {payload['current_model']} "
                    f"({fmt_seconds(payload['elapsed_sec'])} elapsed, "
                    f"~{fmt_seconds(payload['eta_sec'])} remaining)"
                )
                bench = payload['benchmark_df']
                if not bench.empty:
                    fig = px.bar(bench, x='model_name', y='f1_macro', text_auto='.3f',
                                 title='Classifier F1 scores so far')
                    chart_area.plotly_chart(fig, width='stretch')
            elif stage == 'classifier_done':
                progress.progress(50)
                status_box.success(f"Best classifier: {payload['best_model_name']}")
            elif stage == 'topic_progress':
                frac = 50 + int(35 * payload['completed_models'] / payload['total_models'])
                progress.progress(frac)
                timer_box.info(
                    f"Training topic model: {payload['current_model']} "
                    f"({fmt_seconds(payload['elapsed_sec'])} elapsed, "
                    f"~{fmt_seconds(payload['eta_sec'])} remaining)"
                )
                bench = payload['benchmark_df']
                if not bench.empty:
                    fig = px.bar(bench, x='model_name', y='coherence_proxy', text_auto='.3f',
                                 title='Topic model coherence so far')
                    chart_area.plotly_chart(fig, width='stretch')
            elif stage == 'topic_done':
                progress.progress(90)
                sel = payload.get('selection', {})
                summary_box.info(f"Best topic model: {sel.get('selected_model', '—')}")
            elif stage == 'build_done':
                progress.progress(100)
                status_box.success('Training complete!')

        with st.spinner('Training in progress — this may take a few minutes...'):
            summary = build_project_assets(
                BASE, classifier_rows=classifier_rows,
                extra_real_rows=extra_real_rows,
                topic_rows=topic_rows, n_topics=n_topics,
                callback=callback
            )
        st.success('Done! Models saved. Switch to Analyze or Dashboard to see results.')
        with st.expander('Full training summary'):
            st.json(summary)


# Analyze tab

with analyze_tab:
    st.subheader('Analyze an article')
    st.write('Paste a news article to get a fake/real prediction, topic match, and summary.')

    left_col, right_col = st.columns([1, 1], gap='large')

    with left_col:
        title = st.text_input('Headline', placeholder='Paste the article headline here')
        text = st.text_area('Article body', height=300, placeholder='Paste the full article text here...')

        upload = st.file_uploader('Or upload a .txt file', type=['txt'])
        if upload is not None:
            text = upload.read().decode('utf-8', errors='ignore')
            st.text_area('Uploaded text preview', value=text[:500], height=120, disabled=True)

        if not artifacts_ready():
            st.warning('No trained model found. Go to the Train tab first.')

        run_analysis = st.button(
            'Analyze', type='primary',
            width='stretch', disabled=not artifacts_ready()
        )
        if run_analysis:
            st.session_state['analysis_result'] = predict_article(title, text, base_dir=BASE)

    with right_col:
        result = st.session_state.get('analysis_result')
        if not result:
            st.info('Results will appear here after you click Analyze.')
        else:
            pred = result['prediction']
            conf = result['confidence']
            conf_label = confidence_label(conf)

            if pred == 'FAKE':
                st.error(f'**Prediction: {pred}** — Confidence: {conf_label} ({conf:.3f})')
            else:
                st.success(f'**Prediction: {pred}** — Confidence: {conf_label} ({conf:.3f})')

            probs = pd.DataFrame(result['class_probabilities'])
            fig = px.bar(probs, x='label', y='probability', text_auto='.3f',
                         title='Class probabilities', color='label',
                         color_discrete_map={'FAKE': '#e74c3c', 'REAL': '#27ae60'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

            st.divider()
            display_theme = simplify_topic_label(
                result['topic']['topic_name'],
                result['topic'].get('matched_words', [])
            )
            topic_level = topic_confidence_label(
                result['topic']['score'],
                result['topic'].get('confidence_gap', 0.0)
            )
            st.write(f'**Closest topic:** {display_theme} (confidence: {topic_level.lower()})')

            matched = result['topic'].get('matched_words', [])[:5]
            if matched:
                st.write(f'Matched keywords: {", ".join(matched)}')

            topic_df = topic_rows_for_display(result['topic'])
            if not topic_df.empty:
                fig = px.bar(topic_df.sort_values('Score'), x='Score', y='Theme',
                             orientation='h', title='Top topic matches')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')

            st.divider()
            st.write('**Summary:**')
            st.write(result['summary'])

            st.divider()
            st.write('**What drove the prediction?**')
            e1, e2 = st.columns(2)
            fake_terms = pd.DataFrame(result['explanation']['fake_terms'])
            real_terms = pd.DataFrame(result['explanation']['real_terms'])
            with e1:
                st.write('Words pushing toward FAKE:')
                if fake_terms.empty:
                    st.caption('No strong FAKE terms found.')
                else:
                    st.dataframe(fake_terms, width='stretch', hide_index=True)
            with e2:
                st.write('Words pushing toward REAL:')
                if real_terms.empty:
                    st.caption('No strong REAL terms found.')
                else:
                    st.dataframe(real_terms, width='stretch', hide_index=True)

            with st.expander('Raw topic details'):
                st.json({
                    'model_type': result['topic'].get('model_type', 'unknown'),
                    'raw_label': result['topic']['topic_name'],
                    'topic_score': round(result['topic']['score'], 4),
                    'gap_to_next': round(result['topic'].get('confidence_gap', 0.0), 4),
                    'matched_words': result['topic'].get('matched_words', []),
                })


# Verify tab

with verify_tab:
    st.subheader('Fact verification')
    st.write(
        'This mode extracts claims from an article, retrieves evidence from the project evidence corpus, '
        'and assigns a verdict: **Supported**, **Refuted**, or **Not Enough Evidence**.'
    )

    pipe = get_fact_verification_pipeline()
    samples = build_sample_articles_from_raw(BASE)

    left_col, right_col = st.columns([1, 1], gap='large')

    with left_col:
        options = ['(Type your own)']
        if not samples.empty:
            options += samples['headline'].astype(str).tolist()
        picked = st.selectbox('Pick a sample or type your own', options, key='fv_sample')

        if picked != '(Type your own)' and not samples.empty:
            row = samples[samples['headline'].astype(str) == picked].iloc[0]
            default_headline = str(row.get('headline', ''))
            default_text = str(row.get('article_text', ''))
        else:
            default_headline = None
            default_text = None

        # only overwrite text fields when user picks a sample article
        if default_headline is not None:
            st.session_state['fv_headline'] = default_headline
            st.session_state['fv_text'] = default_text

        headline = st.text_input('Headline', key='fv_headline')
        article_text = st.text_area('Article text', height=250, key='fv_text')
        run_verify = st.button('Run verification', type='primary', width='stretch', key='fv_run')

    with right_col:
        if run_verify and (headline.strip() or article_text.strip()):
            result = pipe.analyze_article(headline, article_text)
            claims = result.get('claims', [])

            st.write('**Article summary:**')
            st.write(result.get('summary', 'No summary available.'))
            st.divider()

            if not claims:
                st.info('No strong claims could be extracted from this text.')
            else:
                for idx, claim in enumerate(claims, start=1):
                    verdict = claim.get('verdict', {})
                    label = verdict.get('label', 'NOT ENOUGH EVIDENCE')
                    conf = verdict.get('confidence', 0.0)
                    rationale = verdict.get('rationale', '')

                    st.write(f'**Claim {idx}:** {claim.get("claim", "")}')

                    if label == 'SUPPORTED':
                        st.success(f'Verdict: {label} (confidence: {confidence_label(conf).lower()})')
                    elif label == 'REFUTED':
                        st.error(f'Verdict: {label} (confidence: {confidence_label(conf).lower()})')
                    else:
                        st.warning(f'Verdict: {label} (confidence: {confidence_label(conf).lower()})')

                    st.caption(rationale)

                    evidence = pd.DataFrame(claim.get('evidence', []))
                    if not evidence.empty:
                        ev = evidence[['title', 'source', 'score', 'passage']].copy()
                        ev['score'] = pd.to_numeric(ev['score'], errors='coerce').round(3)
                        with st.expander(f'Evidence passages for claim {idx}', expanded=(idx == 1)):
                            st.dataframe(ev, width='stretch', hide_index=True)

                    if idx < len(claims):
                        st.divider()

        elif run_verify:
            st.warning('Enter a headline or article text first.')
        else:
            st.info('Enter an article and click "Run verification" to see results.')


# Dashboard tab

with dashboard_tab:
    st.subheader('Saved results & benchmarks')

    build = read_json(BASE / 'results' / 'build_summary.json')
    cls_sel = read_json(BASE / 'results' / 'classifier_selection.json')
    topic_sel = read_json(BASE / 'results' / 'topic_selection.json')
    cls_bench = read_csv(BASE / 'results' / 'model_benchmark.csv')
    topic_bench = read_csv(BASE / 'results' / 'topic_model_benchmark.csv')
    perf = read_csv(BASE / 'results' / 'performance_by_source.csv')
    topics = read_csv(BASE / 'results' / 'topic_distribution.csv')
    demos = read_csv(BASE / 'results' / 'pipeline_demo_outputs.csv')
    reps = read_csv(BASE / 'results' / 'topic_representative_titles.csv')

    if build:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Classifier rows', f"{build.get('classifier_rows_used', '—'):,}")
        c2.metric('Best classifier', build.get('best_model', '—'))
        c3.metric('Macro F1', f"{build.get('test_f1_macro', 0):.4f}")
        c4.metric('Topic model', build.get('best_topic_model', '—'))

    dash_tabs = st.tabs(['Overview', 'Classifier results', 'Topic results', 'Example outputs'])

    with dash_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.write('**Classifier selection**')
            if cls_sel:
                st.write(cls_sel.get('selection_reason', 'No info available.'))
                with st.expander('Detailed metrics'):
                    st.json(cls_sel.get('selected_row', {}))
            else:
                st.caption('Run training first to see selection details.')
        with col2:
            st.write('**Topic model selection**')
            if topic_sel:
                st.write(topic_sel.get('selection_reason', 'No info available.'))
                with st.expander('Detailed metrics'):
                    st.json(topic_sel.get('selected_row', {}))
            else:
                st.caption('Run training first to see selection details.')

        if not perf.empty:
            st.write('**Classifier performance by dataset source**')
            fig = px.bar(perf, x='dataset_source', y='f1_macro', text_auto='.3f',
                         title='F1 score on each data source')
            st.plotly_chart(fig, width='stretch')

        if not topics.empty:
            st.write('**Topic distribution**')
            fig = px.bar(topics.sort_values('rows'), x='rows', y='topic_name',
                         orientation='h', title='How many articles per topic')
            st.plotly_chart(fig, width='stretch')

    with dash_tabs[1]:
        if not cls_bench.empty:
            bar_cols = [c for c in ['f1_macro', 'f1_weighted'] if c in cls_bench.columns]
            if bar_cols:
                fig = px.bar(cls_bench, x='model_name', y=bar_cols, barmode='group',
                             title='Classifier comparison')
                st.plotly_chart(fig, width='stretch')

            if 'train_time_sec' in cls_bench.columns:
                hover_cols = [c for c in ['predict_time_ms_per_article', 'accuracy', 'f1_weighted']
                              if c in cls_bench.columns]
                fig = safe_scatter_fig(
                    cls_bench, x='train_time_sec',
                    y='f1_macro' if 'f1_macro' in cls_bench.columns else 'accuracy',
                    color='model_name', hover_data=hover_cols,
                    title='Quality vs training time'
                )
                if fig:
                    st.plotly_chart(fig, width='stretch')

            with st.expander('Full benchmark table'):
                st.dataframe(cls_bench, width='stretch', hide_index=True)
        else:
            st.info('No classifier benchmark data yet. Run training first.')

    with dash_tabs[2]:
        if not topic_bench.empty:
            topic_bar_cols = [c for c in ['quality_score', 'coherence_proxy', 'topic_diversity']
                              if c in topic_bench.columns]
            if topic_bar_cols:
                fig = px.bar(topic_bench, x='model_name', y=topic_bar_cols, barmode='group',
                             title='Topic model comparison')
                st.plotly_chart(fig, width='stretch')

            if 'train_time_sec' in topic_bench.columns:
                hover_cols = [c for c in ['coherence_proxy', 'topic_diversity', 'avg_topic_confidence']
                              if c in topic_bench.columns]
                fig = safe_scatter_fig(
                    topic_bench, x='train_time_sec',
                    y='quality_score' if 'quality_score' in topic_bench.columns else 'coherence_proxy',
                    color='model_name', hover_data=hover_cols,
                    title='Topic quality vs training time'
                )
                if fig:
                    st.plotly_chart(fig, width='stretch')

            with st.expander('Full topic benchmark table'):
                st.dataframe(topic_bench, width='stretch', hide_index=True)
        else:
            st.info('No topic benchmark data yet. Run training first.')

    with dash_tabs[3]:
        if not demos.empty:
            st.write('**Sample predictions from the pipeline**')
            st.dataframe(demos, width='stretch')
        else:
            st.caption('No demo outputs saved yet.')

        if not reps.empty:
            st.write('**Representative article titles per topic**')
            with st.expander('Show titles', expanded=True):
                st.dataframe(reps.head(100), width='stretch')


# Data tab

with data_tab:
    st.subheader('Datasets & setup')

    with st.expander('Required raw data files'):
        st.code("""data/raw/kaggle_fake_real/Fake.csv
data/raw/kaggle_fake_real/True.csv
data/raw/fakenewsnet/BuzzFeed_fake_news_content.csv
data/raw/fakenewsnet/BuzzFeed_real_news_content.csv
data/raw/fakenewsnet/PolitiFact_fake_news_content.csv
data/raw/fakenewsnet/PolitiFact_real_news_content.csv
data/raw/news_category/News_Category_Dataset_v3.json""")
    st.caption('See README.md for full setup instructions.')

    overview = read_json(BASE / 'results' / 'dataset_overview.json')
    if overview:
        with st.expander('Dataset overview (JSON)'):
            st.json(overview)

    cprev = read_csv(BASE / 'data' / 'processed' / 'classification_preview.csv')
    tprev = read_csv(BASE / 'data' / 'processed' / 'topic_preview.csv')
    src = read_csv(BASE / 'results' / 'source_breakdown.csv')
    cats = read_csv(BASE / 'results' / 'topic_input_category_counts.csv')

    col1, col2 = st.columns(2)
    with col1:
        if not src.empty:
            st.write('**Classification data sources**')
            fig = px.pie(src, names='dataset_source', values='rows',
                         title='Where the training data comes from')
            st.plotly_chart(fig, width='stretch')
    with col2:
        if not cats.empty:
            st.write('**Top topic categories in the raw data**')
            fig = px.bar(cats.head(12), x='subject', y='rows',
                         title='Most common news categories')
            st.plotly_chart(fig, width='stretch')

    t1, t2 = st.tabs(['Classification data preview', 'Topic data preview'])
    with t1:
        if not cprev.empty:
            st.dataframe(cprev.head(100), width='stretch')
        else:
            st.caption('No classification preview data available.')
    with t2:
        if not tprev.empty:
            st.dataframe(tprev.head(100), width='stretch')
        else:
            st.caption('No topic preview data available.')

    demo_evidence = read_csv(BASE / 'data' / 'demo' / 'evidence_corpus.csv')
    demo_articles = read_csv(BASE / 'data' / 'demo' / 'sample_articles.csv')

    with st.expander('Fact verification demo corpus'):
        if not demo_evidence.empty:
            st.dataframe(demo_evidence.head(20), width='stretch', hide_index=True)
        else:
            st.caption('Demo evidence corpus not found.')

    with st.expander('Fact verification sample articles'):
        if not demo_articles.empty:
            st.dataframe(demo_articles.head(20), width='stretch', hide_index=True)
        else:
            st.caption('Sample articles not found.')