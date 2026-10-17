"""Explanation stability e_i: SHAP top-k sets under benign perturbations, scored with Jaccard.

Part 1 is the plan's reference implementation (toy-explainer friendly, unit-tested). Part 2 applies it to
the XGBoost model with SHAP TreeExplainer, batched for speed, plus the faithfulness deletion test:
a stable explanation is not automatically a correct one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import shap

# ----------------------------------------------------------------------------------------------
# Part 1 - reference implementation
# ----------------------------------------------------------------------------------------------


def topk_set(attributions, k=5):
    """Indices of the k features with the largest absolute attribution."""
    return set(np.argsort(-np.abs(attributions))[:k].tolist())


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def benign_perturb(x, rng, amount_idx, nonessential_idx, pool, amt_tol=0.02):
    """Small, scenario-preserving change: jitter the amount, resample unimportant fields."""
    z = x.copy()
    z[amount_idx] = z[amount_idx] * (1 + rng.uniform(-amt_tol, amt_tol))
    for j in nonessential_idx:
        z[j] = pool[rng.integers(len(pool)), j]
    return z


def explanation_stability(x, explain_fn, perturb_fn, k=5, M=20, seed=0):
    """e_i = mean Jaccard between original top-k features and top-k after M perturbations."""
    rng = np.random.default_rng(seed)
    base = topk_set(explain_fn(x), k)
    sims = [jaccard(base, topk_set(explain_fn(perturb_fn(x, rng)), k)) for _ in range(M)]
    return float(np.mean(sims))


# ----------------------------------------------------------------------------------------------
# Part 2 - SHAP + vectorised perturbations for the fraud model
# ----------------------------------------------------------------------------------------------


@dataclass
class StabilityResult:
    e: np.ndarray            # (n,) stability per transaction
    shap: np.ndarray         # (n, d) SHAP attributions of the original rows
    p: np.ndarray            # (n,) calibrated probability
    jaccard: np.ndarray      # (n, M)
    prob_shift: np.ndarray   # (n, M)
    kept: np.ndarray         # (n, M) bool - benign perturbations
    topk: np.ndarray         # (n, M, d) bool - top-k membership after each perturbation
    ms_per_alert: float


class StabilityEngine:
    def __init__(self, fitted, calibrator, names: list[str], pool: np.ndarray, cfg: dict):
        self.fitted, self.cal, self.names, self.pool = fitted, calibrator, names, pool
        self.k, self.M = int(cfg["k"]), int(cfg["M"])
        self.amt_tol, self.shift, self.delta = float(cfg["amt_tol"]), float(cfg["time_shift_s"]), float(cfg["delta"])
        self.explainer = shap.TreeExplainer(fitted.model)
        ix = {n: i for i, n in enumerate(names)}
        self.i_amt, self.i_log = ix["amount"], ix["log_amount"]
        self.i_ratio = ix.get("amt_ratio_mean")
        self.i_gap, self.i_hour = ix.get("time_since_prev"), ix["hour"]
        self.nonessential = [i for n, i in ix.items() if n in ("email_domain_te", "card_type_te", "device_type_te")
                             or (n.startswith("v") and n[1:].isdigit())]

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.explainer.shap_values(X, check_additivity=False))

    def prob(self, X: np.ndarray) -> np.ndarray:
        return self.cal(self.fitted.margin(X))

    def _perturb(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        Z = np.tile(x, (self.M, 1))
        f = 1 + rng.uniform(-self.amt_tol, self.amt_tol, self.M)
        Z[:, self.i_amt] *= f
        Z[:, self.i_log] = np.log1p(Z[:, self.i_amt])
        if self.i_ratio is not None:
            Z[:, self.i_ratio] *= f
        dt = rng.uniform(-self.shift, self.shift, self.M)
        if self.i_gap is not None:
            Z[:, self.i_gap] = np.where(x[self.i_gap] >= 1e7, x[self.i_gap], np.maximum(Z[:, self.i_gap] + dt, 0))
        Z[:, self.i_hour] = (Z[:, self.i_hour] + dt / 3600.0) % 24
        if self.nonessential:
            rows = rng.integers(0, len(self.pool), self.M)
            Z[:, self.nonessential] = self.pool[np.ix_(rows, self.nonessential)]
        return Z

    def run(self, X: np.ndarray, seed: int = 0, chunk: int = 60) -> StabilityResult:
        rng = np.random.default_rng(seed)
        n, d, M, k = len(X), X.shape[1], self.M, self.k
        t0 = time.perf_counter()
        sv = self.shap_values(X)
        p = self.prob(X)
        base_mask = np.zeros((n, d), bool)
        base_mask[np.arange(n)[:, None], np.argsort(-np.abs(sv), axis=1)[:, :k]] = True
        jac = np.zeros((n, M)); shift = np.zeros((n, M)); kept = np.zeros((n, M), bool)
        tops = np.zeros((n, M, d), bool)
        for s in range(0, n, chunk):
            idx = range(s, min(s + chunk, n))
            Z = np.concatenate([self._perturb(X[i], rng) for i in idx])
            sz = self.shap_values(Z)
            pz = self.prob(Z)
            m = np.zeros((len(Z), d), bool)
            m[np.arange(len(Z))[:, None], np.argsort(-np.abs(sz), axis=1)[:, :k]] = True
            for j, i in enumerate(idx):
                sl = slice(j * M, (j + 1) * M)
                inter = (m[sl] & base_mask[i]).sum(1); union = (m[sl] | base_mask[i]).sum(1)
                jac[i] = inter / np.maximum(union, 1)
                shift[i] = np.abs(pz[sl] - p[i])
                kept[i] = shift[i] <= self.delta
                tops[i] = m[sl]
        e = np.array([jac[i][kept[i]].mean() if kept[i].any() else jac[i].mean() for i in range(n)])
        return StabilityResult(e, sv, p, jac, shift, kept, tops, (time.perf_counter() - t0) * 1000 / max(n, 1))


def deletion_test(engine: StabilityEngine, X: np.ndarray, sv: np.ndarray, p: np.ndarray,
                  baseline: np.ndarray, seed: int = 0, repeats: int = 5) -> dict[str, float]:
    """Faithfulness: removing the top-k SHAP features should hurt p more than removing k random ones."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    top = np.argsort(-np.abs(sv), axis=1)[:, :engine.k]
    Xd = X.copy()
    Xd[np.arange(n)[:, None], top] = baseline[top]
    drop_top = p - engine.prob(Xd)
    drops = []
    for _ in range(repeats):
        rnd = np.argsort(rng.random((n, d)), axis=1)[:, :engine.k]
        Xr = X.copy()
        Xr[np.arange(n)[:, None], rnd] = baseline[rnd]
        drops.append(p - engine.prob(Xr))
    drop_rand = np.mean(drops, axis=0)
    return {"drop_topk": float(drop_top.mean()), "drop_random": float(drop_rand.mean()),
            "faithful_frac": float((drop_top > drop_rand).mean())}
