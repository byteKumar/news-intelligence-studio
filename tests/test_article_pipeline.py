from pathlib import Path

from src.pipeline import predict_article


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_predict_article_returns_expected_keys() -> None:
    result = predict_article(
        'Sample headline',
        'The government announced a new policy and officials said the change would begin next month.',
        base_dir=PROJECT_ROOT,
    )
    assert {'prediction', 'confidence', 'topic', 'summary', 'explanation'} <= set(result.keys())
    assert result['prediction'] in {'FAKE', 'REAL'}
    assert 0.0 <= result['confidence'] <= 1.0
    assert isinstance(result['summary'], str) and len(result['summary']) > 0
    assert 'topic_name' in result['topic']
