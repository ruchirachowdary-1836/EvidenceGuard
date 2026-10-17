"""Personalised anomaly score a_i in [0, 1] relative to the customer's own past behaviour.

Components (all computed past-only in features.py): robust z of log-amount (median / MAD), hour-of-day
deviation, product rarity. Each is mapped to [0, 1] by an empirical CDF fitted on the TRAINING window and
the three are averaged. Entities with fewer than `cold_start_n` past transactions fall back to a global
IsolationForest trained on the training window (also ECDF-scaled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

GLOBAL_COLS = ["log_amount", "hour", "d1"]


class ECDF:
    def __init__(self, x: np.ndarray):
        self.sorted = np.sort(np.asarray(x, dtype=float))

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.sorted, np.asarray(v, dtype=float), side="right") / max(len(self.sorted), 1)


class AnomalyScorer:
    def __init__(self, cold_start_n: int = 5, seed: int = 0):
        self.n, self.seed = cold_start_n, seed

    def _components(self, f: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.abs(f["z_amt"].to_numpy()), f["hour_dev"].to_numpy(), f["prod_rarity"].to_numpy()])

    def fit(self, train_feat: pd.DataFrame) -> "AnomalyScorer":
        warm = train_feat[train_feat["hist_n"] >= self.n]
        comp = self._components(warm if len(warm) else train_feat)
        self.ecdfs = [ECDF(comp[:, j]) for j in range(comp.shape[1])]
        g = train_feat[GLOBAL_COLS].fillna(0).to_numpy()
        self.iso = IsolationForest(n_estimators=100, random_state=self.seed, n_jobs=1).fit(g)
        self.iso_ecdf = ECDF(-self.iso.score_samples(g))
        return self

    def score(self, f: pd.DataFrame) -> np.ndarray:
        comp = self._components(f)
        personal = np.mean([self.ecdfs[j](comp[:, j]) for j in range(comp.shape[1])], axis=0)
        cold = self.iso_ecdf(-self.iso.score_samples(f[GLOBAL_COLS].fillna(0).to_numpy()))
        return np.where(f["hist_n"].to_numpy() >= self.n, personal, cold).astype(np.float64)
