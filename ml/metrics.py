"""Metrics: PR-AUC (primary), precision / recall @K, bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if np.sum(y) > 0 else float("nan")


def pr_curve(y: np.ndarray, p: np.ndarray, points: int = 60) -> dict[str, list[float]]:
    prec, rec, _ = precision_recall_curve(y, p)
    idx = np.unique(np.linspace(0, len(prec) - 1, points).astype(int))
    return {"precision": prec[idx].round(4).tolist(), "recall": rec[idx].round(4).tolist()}


def precision_recall_at_k(y, chosen):
    tp = float(np.sum(y[np.asarray(chosen, dtype=int)])) if len(chosen) else 0.0
    return tp / max(len(chosen), 1), tp / max(float(np.sum(y)), 1.0)


def bootstrap_ci(values, n=2000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = rng.choice(values, (n, len(values))).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(values.mean()), float(lo), float(hi)


def pr_auc_ci(y, p, n=500, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() > 0:
            vals.append(average_precision_score(y[i], p[i]))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return pr_auc(y, p), float(lo), float(hi)
