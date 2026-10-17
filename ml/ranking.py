"""Combined score R_i, budgeted redundancy-aware queue, per-day evaluation and validation weight search."""
from __future__ import annotations

import heapq
import itertools

import numpy as np

# Variants from the plan: 1 p only | 2 p+a | 3 p+e | 4 all components | 5 full (all + redundancy queue)
VARIANTS = {
    "p_only": dict(use_a=False, use_e=False, redundancy=False),
    "p_a": dict(use_a=True, use_e=False, redundancy=False),
    "p_e": dict(use_a=False, use_e=True, redundancy=False),
    "all": dict(use_a=True, use_e=True, redundancy=False),
    "full": dict(use_a=True, use_e=True, redundancy=True),
}
MODES = ("mult", "linear", "constrained")


def combined_score(p, a, e, mode="mult", alpha=1.0, beta=0.5, gamma=0.5, risk_floor=0.05):
    """p: fraud prob, a: anomaly score in [0,1], e: stability in [0,1] (arrays or scalars).

    linear      R = alpha p + beta a + gamma e   (a stable explanation can lift a low-risk transaction)
    mult        R = p (1 + beta a) (1 - gamma + gamma e)   (cannot rescue a low p)
    constrained like mult, but e only modulates once p >= risk_floor
    """
    p, a, e = np.asarray(p, float), np.asarray(a, float), np.asarray(e, float)
    if mode == "linear":
        return alpha * p + beta * a + gamma * e
    if mode == "mult":
        return p * (1 + beta * a) * (1 - gamma + gamma * e)
    if mode == "constrained":
        stab = np.where(p >= risk_floor, 1 - gamma + gamma * e, 1.0)
        return p * (1 + beta * a) * stab
    raise ValueError(mode)


def select_queue(scores, groups, K, lam=0.3):
    """Lazy greedy: each extra alert from the same group is discounted by (1 - lam). lam=0 -> plain top-K."""
    scores = np.asarray(scores, float)
    counts: dict = {}
    chosen: list[int] = []
    heap = [(-s, i) for i, s in enumerate(scores)]
    heapq.heapify(heap)
    while heap and len(chosen) < K:
        _, i = heapq.heappop(heap)
        gain = scores[i] * (1 - lam) ** counts.get(groups[i], 0)
        if heap and gain < -heap[0][0]:
            heapq.heappush(heap, (-gain, i))
            continue
        chosen.append(i)
        counts[groups[i]] = counts.get(groups[i], 0) + 1
    return chosen


def redundancy(groups, chosen) -> float:
    """Share of queued alerts that have another alert from the same cluster in the same queue."""
    g = [groups[i] for i in chosen]
    if not g:
        return 0.0
    vals, cnt = np.unique(g, return_counts=True)
    dup = dict(zip(vals.tolist(), cnt.tolist()))
    return float(sum(1 for x in g if dup[x] > 1) / len(g))


def rank_day(p, a, e, groups, K, *, variant="full", mode="mult", beta=0.5, gamma=0.5, lam=0.3,
             alpha=1.0, risk_floor=0.05):
    """Queue (indices into the day's candidate arrays) for one day under one method."""
    v = VARIANTS[variant]
    b = beta if v["use_a"] else 0.0
    g = gamma if v["use_e"] else 0.0
    if variant == "p_only":
        score = np.asarray(p, float)
    else:
        score = combined_score(p, a, e, mode, alpha, b, g, risk_floor)
    return select_queue(score, groups, K, lam if v["redundancy"] else 0.0), score


def evaluate(days: list[dict], budget_frac: float | None = None, k_abs: int | None = None, total_fraud: float | None = None,
             **kw):
    """days: dicts with arrays p,a,e,groups,y and n_day (all txns that day). Pooled recall/precision.

    Review capacity per day is K = ceil(budget_frac * n_day) (or k_abs). Recall is over ALL fraud in the
    window (including fraud outside the candidate pool), so the pool can only cost recall, never hide it.
    """
    tp = queued = 0.0
    hits, dups = [], []
    for d in days:
        K = k_abs if k_abs is not None else max(1, int(np.ceil(budget_frac * d["n_day"])))
        K = min(K, len(d["p"]))
        chosen, _ = rank_day(d["p"], d["a"], d["e"], d["groups"], K, **kw)
        ch = np.asarray(chosen, int)
        tp += float(d["y"][ch].sum()); queued += len(ch)
        hits.append(d["y"][ch]); dups.append(redundancy(d["groups"], chosen) * len(ch))
    total = total_fraud if total_fraud is not None else float(sum(d["y_total"] for d in days))
    return {"recall": tp / max(total, 1.0), "precision": tp / max(queued, 1.0), "tp": tp, "queued": queued,
            "total_fraud": total, "redundancy": float(sum(dups) / max(queued, 1.0)), "hits": hits}


def search_weights(days: list[dict], variant: str, mode: str, grid: dict, budget_frac: float, risk_floor: float,
                   total_fraud: float | None = None):
    """Grid search on VALIDATION days only; objective = recall at the review budget (ties -> lower precision loss)."""
    v = VARIANTS[variant]
    betas = grid["beta"] if v["use_a"] else [0.0]
    gammas = grid["gamma"] if v["use_e"] else [0.0]
    lams = grid["lam"] if v["redundancy"] else [0.0]
    best, best_key = None, (-1.0, -1.0)
    for beta, gamma, lam in itertools.product(betas, gammas, lams):
        r = evaluate(days, budget_frac, total_fraud=total_fraud, variant=variant, mode=mode, beta=beta, gamma=gamma,
                     lam=lam, risk_floor=risk_floor)
        key = (round(r["recall"], 6), r["precision"])
        if key > best_key:
            best_key, best = key, {"beta": beta, "gamma": gamma, "lam": lam, "val_recall": r["recall"]}
    return best
