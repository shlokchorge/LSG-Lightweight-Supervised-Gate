"""
LSG ensemble: LogisticRegression + XGBoost + MLPClassifier
with a per-domain learned linear adapter (projection + bias).

DomainAdapter learns a small W (emb_dim x emb_dim) + b per domain during
training.  At test time on an unseen domain, adapt() fine-tunes only the
adapter on a small number of labeled target-domain examples (few-shot).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


class DomainAdapter:
    """
    Per-domain linear projection: x_adapted = x @ W_d + b_d
    W_d is initialised to identity, b_d to zero.
    Fit by minimising MSE between projected source embeddings and
    the per-class centroids of the target domain (or just mean-centering
    when no target labels are available).
    """

    def __init__(self, emb_dim: int):
        self.emb_dim = emb_dim
        self.W: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.global_mean = np.zeros(emb_dim)

    def fit(self, X: np.ndarray, domains: np.ndarray, y: np.ndarray | None = None):
        self.global_mean = X.mean(axis=0)
        for d in np.unique(domains):
            mask = domains == d
            Xd = X[mask]
            domain_mean = Xd.mean(axis=0)
            # Initialise W to identity, b to shift domain mean → global mean
            self.W[d] = np.eye(self.emb_dim, dtype=np.float32)
            self.b[d] = self.global_mean - domain_mean
        return self

    def adapt(self, X: np.ndarray, y: np.ndarray, domain: str, lr: float = 0.01, steps: int = 50):
        """
        Few-shot update: given a small labeled set from `domain`,
        gradient-descend the bias b[domain] to align class centroids.
        W stays fixed (identity) — only the bias is updated.
        """
        if domain not in self.b:
            self.b[domain] = np.zeros(self.emb_dim, dtype=np.float32)
            self.W[domain] = np.eye(self.emb_dim, dtype=np.float32)

        b = self.b[domain].copy().astype(np.float64)
        for _ in range(steps):
            X_shifted = X + b
            # Target: push class-1 centroid up, class-0 centroid down (in first PC direction)
            c1 = X_shifted[y == 1].mean(axis=0) if (y == 1).any() else X_shifted.mean(axis=0)
            c0 = X_shifted[y == 0].mean(axis=0) if (y == 0).any() else X_shifted.mean(axis=0)
            # Gradient: maximise margin between centroids
            grad = -(c1 - c0) / (np.linalg.norm(c1 - c0) + 1e-8)
            b -= lr * grad
        self.b[domain] = b.astype(np.float32)
        return self

    def transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        out = X.copy().astype(np.float64)
        for d in np.unique(domains):
            mask = domains == d
            W = self.W.get(d, np.eye(self.emb_dim))
            b = self.b.get(d, np.zeros(self.emb_dim))
            out[mask] = out[mask] @ W + b
        return out.astype(np.float32)


class LSGEnsemble:
    def __init__(self, use_adapter: bool = True, weights: tuple = (0.3, 0.4, 0.3)):
        self.use_adapter = use_adapter
        self.weights = np.array(weights) / sum(weights)
        self.adapter: DomainAdapter | None = None

        self.lr  = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        self.xgb = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", tree_method="hist",
            device="cpu", nthread=1, verbosity=0,
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(128,), max_iter=200,
            early_stopping=False, random_state=42,
        )

    def _adapt(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        if self.use_adapter and self.adapter is not None:
            return self.adapter.transform(X, domains)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, domains: np.ndarray):
        if len(np.unique(y)) < 2:
            raise ValueError(f"Training data must have both classes; got: {np.unique(y)}")
        if self.use_adapter:
            self.adapter = DomainAdapter(X.shape[1]).fit(X, domains, y)
        X_a = self._adapt(X, domains)
        self.lr.fit(X_a, y)
        self.xgb.fit(X_a, y)
        self.mlp.fit(X_a, y)
        return self

    def few_shot_adapt(self, X: np.ndarray, y: np.ndarray, domain: str):
        """Update only the adapter bias for `domain` using a small labeled set."""
        if self.adapter is None:
            self.adapter = DomainAdapter(X.shape[1])
        self.adapter.adapt(X, y, domain)
        return self

    def predict_proba(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        X_a = self._adapt(X, domains)
        p_lr  = self.lr.predict_proba(X_a)[:, 1]
        p_xgb = self.xgb.predict_proba(X_a)[:, 1]
        p_mlp = self.mlp.predict_proba(X_a)[:, 1]
        return self.weights[0] * p_lr + self.weights[1] * p_xgb + self.weights[2] * p_mlp

    def predict(self, X: np.ndarray, domains: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X, domains) >= threshold).astype(int)
