"""Top-level article analysis pipeline for classification, topics, and summary."""

from __future__ import annotations
from pathlib import Path
from .classifier import predict_details
from .summarizer import summarize
from .topic_inference import infer_topic


def predict_article(title: str, text: str, base_dir: Path | None = None) -> dict:
    """Run the end-to-end article analysis pipeline.

    Args:
        title: Article headline.
        text: Article body text.
        base_dir: Optional project root for loading artifacts.

    Returns:
        Dict containing classifier result, topic inference, summary, and
        token-level explanation terms.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[1]
    combined = f"{str(title or '').strip()} {str(text or '').strip()}".strip()
    clf = predict_details(combined, base_dir=base_dir)
    topic = infer_topic(combined, base_dir=base_dir)
    summary = summarize(text or combined, max_sentences=3)
    return {
        'prediction': clf['prediction'],
        'confidence': clf['confidence'],
        'margin': clf['margin'],
        'class_probabilities': clf['class_probabilities'],
        'topic': topic,
        'summary': summary,
        'explanation': clf['explanation'],
    }
