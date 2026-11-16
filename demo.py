"""Standalone full execution script for Team 116 News Intelligence Studio.

This is the single terminal entrypoint for the project. It always rebuilds the
project from the bundled raw datasets in ``data/raw/``, saves fresh outputs to
``results/`` and then runs both the analyzer and verifier on one article.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fact_verify.loaders import build_sample_articles_from_raw
from src.notebook_runner import analyze_and_save_custom_article, run_full_rebuild


def run_project(
    base_dir: str | Path | None = None,
    *,
    headline: str | None = None,
    article_text: str | None = None,
    n_topics: int = 12,
    extra_real_rows: int = 2000,
    topic_rows: int = 12000,
) -> dict[str, Any]:
    """Run the full project from raw data and return saved outputs."""
    project_root = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
    rebuild_summary = run_full_rebuild(
        project_root,
        n_topics=n_topics,
        extra_real_rows=extra_real_rows,
        topic_rows=topic_rows,
    )

    if not headline or not article_text:
        sample_articles = build_sample_articles_from_raw(project_root)
        sample_row = sample_articles.iloc[0]
        headline = str(sample_row['headline'])
        article_text = str(sample_row['article_text'])

    custom_result = analyze_and_save_custom_article(project_root, headline, article_text)
    return {
        'rebuild_summary': rebuild_summary,
        'headline': headline,
        'article_text': article_text,
        'custom_result': custom_result,
        'saved_result_path': str(project_root / 'results' / 'custom_article_result.json'),
    }


def print_terminal_summary(result: dict[str, Any]) -> None:
    """Print a compact human-readable summary for terminal runs."""
    print('=== Team 116 News Intelligence Studio ===')
    print('Project rebuilt from raw data and fresh outputs saved to results/.')
    print(json.dumps(result['rebuild_summary'], indent=2))
    print('\n[Sample article used]')
    print('Headline:', result['headline'])
    print('\n[Analyzer]')
    analysis = result['custom_result']['analysis']
    print('Prediction:', analysis['prediction'])
    print('Confidence:', round(analysis['confidence'], 4))
    print('Closest theme:', analysis['topic']['topic_name'])
    print('Summary:', analysis['summary'])
    print('\n[Verifier]')
    verification = result['custom_result']['verification']
    print('Verification summary:', verification['summary'])
    for index, claim in enumerate(verification['claims'], start=1):
        print(f'Claim {index}:', claim['claim'])
        print('Verdict:', claim['verdict']['label'])
        print('Rationale:', claim['verdict']['rationale'])
        if claim['evidence']:
            top_hit = claim['evidence'][0]
            print('Top evidence title:', top_hit['title'])
            print('Top evidence score:', round(top_hit['score'], 4))
        print('-' * 60)
    print('\nSaved combined article result to', result['saved_result_path'])


def main() -> None:
    """Execute the full standalone project run from the command line."""
    result = run_project()
    print_terminal_summary(result)


if __name__ == '__main__':
    main()
