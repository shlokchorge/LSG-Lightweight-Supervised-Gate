"""Evaluation metrics: P/R/F1, ROC-AUC, ECE, and timing."""
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


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
    ece = expected_calibration_error(y_true, probs)
    return {"precision": p, "recall": r, "f1": f1, "roc_auc": auc, "ece": ece}


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
