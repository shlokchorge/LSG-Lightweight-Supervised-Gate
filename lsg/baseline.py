"""
Statistical baseline gate.
Decision rule: STORE if max cosine-sim to existing memory < novelty_thresh
               AND recency score > recency_thresh.
No training required.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class StatisticalGate:
    def __init__(self, novelty_thresh: float = 0.85, recency_decay: float = 0.95):
        self.novelty_thresh = novelty_thresh
        self.recency_decay = recency_decay

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Simulate a streaming memory: for each sample in order, decide
        STORE(1) or IGNORE(0) based on novelty vs existing memory.
        """
        memory: list[np.ndarray] = []
        recency_weights: list[float] = []
        preds = np.zeros(len(embeddings), dtype=int)

        for i, emb in enumerate(embeddings):
            emb = emb.reshape(1, -1)
            if not memory:
                preds[i] = 1
                memory.append(emb)
                recency_weights.append(1.0)
                continue

            mem_matrix = np.vstack(memory)
            sims = cosine_similarity(emb, mem_matrix)[0]
            # Weight similarities by recency
            weights = np.array(recency_weights)
            weighted_sim = np.max(sims * weights)

            if weighted_sim < self.novelty_thresh:
                preds[i] = 1
                memory.append(emb)
                recency_weights.append(1.0)
            else:
                preds[i] = 0

            # Decay recency weights
            recency_weights = [w * self.recency_decay for w in recency_weights]

        return preds

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Return soft scores (novelty score) for AUC/calibration."""
        memory: list[np.ndarray] = []
        recency_weights: list[float] = []
        scores = np.zeros(len(embeddings))

        for i, emb in enumerate(embeddings):
            emb = emb.reshape(1, -1)
            if not memory:
                scores[i] = 1.0
                memory.append(emb)
                recency_weights.append(1.0)
                continue

            mem_matrix = np.vstack(memory)
            sims = cosine_similarity(emb, mem_matrix)[0]
            weights = np.array(recency_weights)
            weighted_sim = np.max(sims * weights)
            novelty = 1.0 - weighted_sim
            scores[i] = novelty

            if novelty > (1.0 - self.novelty_thresh):
                memory.append(emb)
                recency_weights.append(1.0)

            recency_weights = [w * self.recency_decay for w in recency_weights]

        return scores
