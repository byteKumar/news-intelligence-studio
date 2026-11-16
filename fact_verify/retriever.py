"""TF-IDF evidence retriever used in the verification baseline."""

from dataclasses import dataclass
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvidenceHit:
    """Single retrieved evidence row with retrieval similarity score."""
    doc_id: str
    title: str
    passage: str
    source: str
    score: float


class TfidfRetriever:
    """Retrieve top-k evidence passages by TF-IDF cosine similarity.

    The retriever vectorizes each evidence document using title + passage text.
    At query time, a claim is transformed into the same TF-IDF space and ranked
    against all evidence documents by cosine similarity.
    """

    def __init__(self, corpus_df: pd.DataFrame):
        """Build retrieval index from an evidence corpus DataFrame."""
        self.corpus_df = corpus_df.reset_index(drop=True).copy()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        self.doc_vectors = self.vectorizer.fit_transform(
            (self.corpus_df["title"].fillna("") + " " + self.corpus_df["passage"].fillna("")).tolist()
        )

    def retrieve(self, claim_text: str, top_k: int = 3) -> List[EvidenceHit]:
        """Return top-k evidence hits sorted by similarity (highest first)."""
        q = self.vectorizer.transform([claim_text])
        sims = cosine_similarity(q, self.doc_vectors).ravel()
        top_idx = sims.argsort()[::-1][:top_k]
        hits: List[EvidenceHit] = []
        for idx in top_idx:
            row = self.corpus_df.iloc[int(idx)]
            hits.append(EvidenceHit(
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                passage=str(row["passage"]),
                source=str(row.get("source", "demo_corpus")),
                score=float(sims[idx]),
            ))
        return hits
