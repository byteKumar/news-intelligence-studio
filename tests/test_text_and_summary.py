from src.text_utils import normalize_text
from src.summarizer import summarize


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text('Hello\n\tworld   !') == 'hello world'


def test_summarize_returns_at_most_requested_sentences() -> None:
    text = (
        'Sentence one talks about policy. '
        'Sentence two adds more detail about elections. '
        'Sentence three mentions public reaction. '
        'Sentence four closes the article.'
    )
    summary = summarize(text, max_sentences=2)
    sentence_count = summary.count('.')
    assert sentence_count <= 2
    assert 'Sentence' in summary
