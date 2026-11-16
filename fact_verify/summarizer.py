"""Extractive summarizer for the fact-verification pipeline.

Delegates to the shared implementation in src.summarizer to avoid
duplicating the frequency-based scoring logic.
"""

from src.summarizer import summarize_basic as summarize_article

__all__ = ['summarize_article']
