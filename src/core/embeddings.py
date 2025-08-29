"""
Lightweight text embeddings using TF-IDF (local, no external API),
with cosine, L2 (Euclidean), and dot-product similarities.
Useful to retrieve similar past queries before calling the LLM.
"""

from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class SimpleEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None
        self.corpus: List[str] = []

    def fit(self, texts: List[str]):
        self.corpus = texts[:]
        self.matrix = self.vectorizer.fit_transform(
            texts)  # (n_docs, n_features)

    def encode(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    @staticmethod
    def cosine_sim(a, b) -> float:
        # a,b are 1xN sparse vectors
        denom = np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray())
        if denom == 0:
            return 0.0
        return float(a.multiply(b).sum() / denom)

    @staticmethod
    def dot_product(a, b) -> float:
        return float(a.multiply(b).sum())

    @staticmethod
    def l2_distance(a, b) -> float:
        diff = a.toarray() - b.toarray()
        return float(np.linalg.norm(diff))

    def most_similar(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        qv = self.encode([query])
        sims = []
        for i in range(self.matrix.shape[0]):
            docv = self.matrix.getrow(i)
            sims.append((i, self.cosine_sim(qv, docv)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]
