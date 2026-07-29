"""Evaluation metrics: P/R/F1, ROC-AUC, AUPRC, ECE, bootstrap CI, and timing."""
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = probs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return ece


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    try:
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        auprc = float("nan")
    ece = expected_calibration_error(y_true, probs)
    return {"precision": p, "recall": r, "f1": f1, "roc_auc": auc, "auprc": auprc, "ece": ece}


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, metric: str = "auprc",
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """
    Bootstrap confidence interval for a scalar metric.
    metric: one of 'auprc', 'roc_auc', 'f1'
    Returns (lower, upper) at (alpha/2, 1-alpha/2) percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "auprc":
            stats.append(average_precision_score(yt, yp))
        elif metric == "roc_auc":
            stats.append(roc_auc_score(yt, yp))
        elif metric == "f1":
            preds = (yp >= 0.5).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(yt, preds, pos_label=1, average="binary", zero_division=0)
            stats.append(f1)
    stats = np.array(stats)
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
