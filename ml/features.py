"""Leakage-safe behavioural features.

Rule: a feature for transaction t may only use rows with time STRICTLY earlier than t (ties excluded),
so every window / history statistic below is computed from sorted, shifted prefixes. The unit tests
recompute the features on a truncated dataset and require identical values (no future information).

Note on the entity proxy: entity_key already contains the address (card1 + addr1), so an address can
never be "new" for an entity; the flags are therefore new_device and new_email.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400
BIG_GAP = 1e7


def _prefix(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])


def _windows(t: np.ndarray, start: np.ndarray, w: int) -> np.ndarray:
    """Number of strictly-earlier rows with time >= t - w (rows are sorted by time)."""
    return start - np.searchsorted(t, t - w, side="left")


def build_features(df: pd.DataFrame, windows_s: list[int], history_cap: int = 50) -> pd.DataFrame:
    n = len(df)
    ts = df["ts"].to_numpy(np.int64)
    amount = df["amount"].to_numpy(float)
    lamt = np.log1p(amount)
    hour = (ts % DAY) / 3600.0
    ang = hour / 24.0 * 2 * np.pi
    prod = df["product"].to_numpy(object)
    dev = df["cluster_key"].to_numpy(object)
    email = df["email_domain"].to_numpy(object)

    out = {k: np.zeros(n) for k in
           [f"cnt_{w}" for w in windows_s] + ["sum_3600", "sum_86400", "hist_n", "mean_amt", "median_amt", "std_amt",
                                              "time_since_prev", "z_amt", "hour_dev", "prod_rarity", "new_device",
                                              "new_email", "amt_ratio_mean", "cluster_cnt_3600", "cluster_cnt_86400"]}
    out["time_since_prev"][:] = BIG_GAP

    codes, _ = pd.factorize(df["entity_key"])
    order = np.argsort(codes, kind="mergesort")           # stable: keeps time order inside each entity
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for g in np.split(order, bounds):
        t, a, la = ts[g], amount[g], lamt[g]
        start = np.searchsorted(t, t, side="left")        # rows strictly earlier than each row
        P1, P2, PL = _prefix(a), _prefix(a * a), _prefix(la)
        Ps, Pc = _prefix(np.sin(ang[g])), _prefix(np.cos(ang[g]))
        for w in windows_s:
            out[f"cnt_{w}"][g] = _windows(t, start, w)
        for w, key in ((3600, "sum_3600"), (86400, "sum_86400")):
            lo = np.searchsorted(t, t - w, side="left")
            out[key][g] = P1[start] - P1[lo]
        hn = start.astype(float)
        out["hist_n"][g] = hn
        nz = np.maximum(hn, 1)
        mean = P1[start] / nz
        var = np.maximum(P2[start] / nz - mean ** 2, 0)
        out["mean_amt"][g] = np.where(hn > 0, mean, 0)
        out["std_amt"][g] = np.where(hn > 1, np.sqrt(var), 0)
        out["amt_ratio_mean"][g] = np.where(hn > 0, a / (mean + 1.0), 1.0)
        prev = np.where(start > 0, t - t[np.maximum(start - 1, 0)], BIG_GAP)
        out["time_since_prev"][g] = prev
        # circular mean hour of strictly-past rows
        s_sum, c_sum = Ps[start], Pc[start]
        mu = np.arctan2(s_sum, c_sum)
        d = np.abs((ang[g] - mu + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi) * 24
        out["hour_dev"][g] = np.where(hn > 0, d, 0)
        # robust z of log-amount vs last `history_cap` past rows; product rarity; new device / email
        z = np.zeros(len(g)); med = np.zeros(len(g)); rar = np.zeros(len(g))
        seen_prod: dict = {}; seen_dev: set = set(); seen_email: set = set()
        nd = np.zeros(len(g)); ne = np.zeros(len(g))
        ptr = 0
        gp, gd, ge = prod[g], dev[g], email[g]
        for i in range(len(g)):
            while ptr < start[i]:                         # fold in rows that are strictly earlier
                seen_prod[gp[ptr]] = seen_prod.get(gp[ptr], 0) + 1
                seen_dev.add(gd[ptr]); seen_email.add(ge[ptr]); ptr += 1
            if start[i] > 0:
                h = la[max(0, start[i] - history_cap):start[i]]
                m = float(np.median(h)); mad = float(np.median(np.abs(h - m)))
                med[i] = np.expm1(m)
                z[i] = np.clip((la[i] - m) / (1.4826 * mad + 0.15), -10, 10)
                rar[i] = 1.0 - seen_prod.get(gp[i], 0) / start[i]
                nd[i] = float(gd[i] not in seen_dev); ne[i] = float(ge[i] not in seen_email)
        out["z_amt"][g], out["median_amt"][g], out["prod_rarity"][g] = z, med, rar
        out["new_device"][g], out["new_email"][g] = nd, ne

    # cluster-level velocity (past-only counts per device group, any entity)
    ccodes, _ = pd.factorize(df["cluster_key"])
    corder = np.argsort(ccodes, kind="mergesort")
    for g in np.split(corder, np.flatnonzero(np.diff(ccodes[corder])) + 1):
        t = ts[g]
        start = np.searchsorted(t, t, side="left")
        out["cluster_cnt_3600"][g] = _windows(t, start, 3600)
        out["cluster_cnt_86400"][g] = _windows(t, start, 86400)

    feat = df.copy()
    for k, v in out.items():
        feat[k] = v
    feat["log_amount"] = lamt
    feat["hour"] = hour
    feat["weekday"] = (ts // DAY) % 7
    for c in df.columns:
        if c.startswith("v") and c[1:].isdigit():
            feat[c] = df[c].to_numpy(float)
    return feat


CAT_COLS = ["product", "card_type", "email_domain", "device_type"]
HISTORY_COLS = ["cnt_3600", "cnt_86400", "cnt_604800", "cnt_2592000", "sum_3600", "sum_86400", "hist_n", "mean_amt",
                "median_amt", "std_amt", "time_since_prev", "z_amt", "hour_dev", "prod_rarity", "new_device",
                "new_email", "amt_ratio_mean", "cluster_cnt_3600", "cluster_cnt_86400"]
RAW_COLS = ["amount", "log_amount", "d1", "hour", "weekday"]


def numeric_columns(feat: pd.DataFrame) -> list[str]:
    extras = sorted(c for c in feat.columns if c.startswith("v") and c[1:].isdigit())
    hist = [c for c in HISTORY_COLS if c in feat.columns]
    return RAW_COLS + extras + hist


# Fields a *benign* perturbation may resample: they carry no scenario meaning for the transaction.
NONESSENTIAL = ["email_domain_te", "card_type_te", "device_type_te"]
