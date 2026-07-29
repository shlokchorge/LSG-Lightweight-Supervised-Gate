"""
LSG ensemble: LogisticRegression + XGBoost + MLPClassifier
with a per-domain learned linear adapter (projection + bias).

DomainAdapter learns a per-domain bias b and a rank-1 W correction.
few_shot_adapt() updates both b and W, then recalibrates the decision
threshold on the few-shot labels — the only place we have target-domain
ground truth.

fit() trains on the full source data with threshold=0.5 (no source-domain
threshold leakage to target).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


class DomainAdapter:
    def __init__(self, emb_dim: int):
        self.emb_dim = emb_dim
        self.W: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.global_mean = np.zeros(emb_dim)

    def fit(self, X: np.ndarray, domains: np.ndarray, y: np.ndarray | None = None):
        self.global_mean = X.mean(axis=0)
        for d in np.unique(domains):
            mask = domains == d
            self.W[d] = np.eye(self.emb_dim, dtype=np.float32)
            self.b[d] = (self.global_mean - X[mask].mean(axis=0)).astype(np.float32)
        return self

    def adapt(self, X: np.ndarray, y: np.ndarray, domain: str,
              lr: float = 0.05, steps: int = 100):
        """Gradient-descend b to maximise centroid margin + rank-1 W correction."""
        if domain not in self.b:
            self.b[domain] = np.zeros(self.emb_dim, dtype=np.float32)
            self.W[domain] = np.eye(self.emb_dim, dtype=np.float32)

        b = self.b[domain].copy().astype(np.float64)
        W = self.W[domain].copy().astype(np.float64)

        for step in range(steps):
            X_shifted = X @ W.T + b
            c1 = X_shifted[y == 1].mean(axis=0) if (y == 1).any() else X_shifted.mean(axis=0)
            c0 = X_shifted[y == 0].mean(axis=0) if (y == 0).any() else X_shifted.mean(axis=0)
            diff = c1 - c0
            norm = np.linalg.norm(diff) + 1e-8
            b -= lr * (-diff / norm)
            if step % 20 == 0 and norm > 1e-4:
                v = diff / norm
                W += min(0.05, lr) * np.outer(v, v)

        self.b[domain] = b.astype(np.float32)
        self.W[domain] = W.astype(np.float32)
        return self

    def transform(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        out = X.copy().astype(np.float64)
        for d in np.unique(domains):
            mask = domains == d
            W = self.W.get(d, np.eye(self.emb_dim))
            b = self.b.get(d, np.zeros(self.emb_dim))
            out[mask] = out[mask] @ W.T + b
        return out.astype(np.float32)


class LSGEnsemble:
    def __init__(self, use_adapter: bool = True, weights: tuple = (0.25, 0.50, 0.25)):
        self.use_adapter = use_adapter
        self.weights = np.array(weights) / sum(weights)
        self.adapter: DomainAdapter | None = None
        self.threshold: float = 0.5

        self.lr = LogisticRegression(
            max_iter=2000, C=0.5, solver="lbfgs", class_weight="balanced"
        )
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            eval_metric="logloss", tree_method="hist",
            device="cpu", nthread=1, verbosity=0,
        )
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 64), max_iter=400,
            learning_rate_init=0.001, alpha=1e-3,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        )

    def _adapt(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        if self.use_adapter and self.adapter is not None:
            return self.adapter.transform(X, domains)
        return X

    def _raw_proba(self, X_a: np.ndarray) -> np.ndarray:
        p_lr  = self.lr.predict_proba(X_a)[:, 1]
        p_xgb = self.xgb.predict_proba(X_a)[:, 1]
        p_mlp = self.mlp.predict_proba(X_a)[:, 1]
        return self.weights[0] * p_lr + self.weights[1] * p_xgb + self.weights[2] * p_mlp

    def fit(self, X: np.ndarray, y: np.ndarray, domains: np.ndarray):
        if len(np.unique(y)) < 2:
            raise ValueError(f"Training data must have both classes; got: {np.unique(y)}")
        X_np = np.asarray(X)
        y_np = np.asarray(y)
        d_np = np.asarray(domains)
        if self.use_adapter:
            self.adapter = DomainAdapter(X_np.shape[1]).fit(X_np, d_np, y_np)
        X_a = self._adapt(X_np, d_np)
        self.lr.fit(X_a, y_np)
        self.xgb.fit(X_a, y_np)
        self.mlp.fit(X_a, y_np)
        self.threshold = 0.5   # reset; calibrated only when target labels available
        return self

    def few_shot_adapt(self, X: np.ndarray, y: np.ndarray, domain: str):
        """Update adapter for `domain`, then calibrate threshold on the few-shot labels."""
        if self.adapter is None:
            self.adapter = DomainAdapter(X.shape[1])
        self.adapter.adapt(X, y, domain)
        # Calibrate threshold using the few-shot target labels
        d_arr = np.array([domain] * len(X))
        probs = self._raw_proba(self._adapt(X, d_arr))
        if len(np.unique(y)) == 2:
            prec, rec, thresholds = precision_recall_curve(y, probs)
            f1s = 2 * prec * rec / (prec + rec + 1e-9)
            self.threshold = float(thresholds[f1s[:-1].argmax()])
        return self

    def predict_proba(self, X: np.ndarray, domains: np.ndarray) -> np.ndarray:
        return self._raw_proba(self._adapt(X, domains))

    def predict(self, X: np.ndarray, domains: np.ndarray, threshold: float | None = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X, domains) >= thr).astype(int)
