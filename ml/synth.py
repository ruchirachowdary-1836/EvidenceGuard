"""Synthetic card-not-present transactions with IEEE-CIS-like columns and coordinated fraud rings.

This exists so the whole pipeline (and the demo database) runs without Kaggle access. It is a simulator
with *our own* assumptions - results on it demonstrate that the pipeline works, not that the method
detects real fraud. Never quote them as findings about IEEE-CIS, PaySim or UPI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = np.array(["W", "C", "H", "R", "S"])
EMAILS = np.array(["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "anonymous.com", "proton.me"])
CARD_TYPES = np.array(["credit", "debit"])
DEVICES = np.array(["desktop", "mobile", "tablet"])
DAY = 86400


def generate(n_entities: int = 1400, n_days: int = 60, n_rings: int = 85, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    horizon = n_days * DAY

    # ---- entities (a pseudo-customer is card1 + addr1, as in the IEEE-CIS entity proxy) ----
    card1 = rng.choice(np.arange(1000, 1000 + n_entities * 4), n_entities, replace=False)
    addr1 = rng.integers(100, 500, n_entities)
    log_med = rng.normal(3.6, 0.7, n_entities)
    log_sd = rng.uniform(0.35, 0.7, n_entities)
    hour_mu = rng.normal(14, 3.5, n_entities)
    hour_sd = rng.uniform(2.5, 5.0, n_entities)
    pref_prod = rng.choice(PRODUCTS, n_entities, p=[.55, .15, .1, .1, .1])
    email = rng.choice(EMAILS, n_entities, p=[.4, .2, .15, .1, .05, .1])
    ctype = rng.choice(CARD_TYPES, n_entities, p=[.35, .65])
    age0 = rng.integers(5, 900, n_entities)
    dev_ids = [[f"dev{e}a", f"dev{e}b"] for e in range(n_entities)]
    for e in rng.choice(n_entities, n_entities // 12, replace=False):   # shared household devices
        dev_ids[e][0] = f"dev{rng.integers(0, n_entities)}a"
    rate = rng.gamma(1.4, 0.55, n_entities) * (0.6 + 0.0)               # txns per day, heavy tailed

    rows = []

    def noise_v():
        return rng.normal(size=6)

    # ---- legitimate behaviour ----
    for e in range(n_entities):
        n = max(3, rng.poisson(rate[e] * n_days))
        ts = np.sort(rng.integers(0, horizon, n))
        h = np.clip(rng.normal(hour_mu[e], hour_sd[e], n), 0, 23.99)
        ts = (ts // DAY) * DAY + (h * 3600 + rng.integers(0, 600, n)).astype(int)
        amt = np.exp(rng.normal(log_med[e], log_sd[e], n)).round(2)
        prod = np.where(rng.random(n) < .82, pref_prod[e], rng.choice(PRODUCTS, n))
        dev = np.where(rng.random(n) < .94, dev_ids[e][0], dev_ids[e][1])
        for i in range(n):
            v = noise_v()
            a_i, d_i = amt[i], dev[i]
            if rng.random() < .05:                                  # hard negative: legitimate big purchase
                a_i = round(float(a_i * rng.uniform(3, 7)), 2)
            if rng.random() < .06:                                  # hard negative: new phone / travel
                d_i = f"devnew{e}_{i}"
            rows.append((ts[i], card1[e], addr1[e], a_i, prod[i], ctype[e], email[e],
                         rng.choice(DEVICES, p=[.45, .45, .1]), d_i, age0[e] + ts[i] / DAY, 0, *v))

    # ---- fraud rings: one attacker device, several victim cards, bursts of activity ----
    for r in range(n_rings):
        ring_dev = f"ringdev{r}"
        victims = rng.choice(n_entities, rng.integers(3, 11), replace=False)
        start = rng.integers(2 * DAY, horizon - DAY)
        ring_email = rng.choice(["anonymous.com", "proton.me", "gmail.com"], p=[.45, .35, .2])
        night = rng.random() < .35
        for e in victims:
            t0 = start + rng.integers(0, 6 * 3600)
            for _ in range(rng.integers(1, 5)):
                t = t0 + rng.integers(0, 12 * 3600)
                if night and rng.random() < .5:
                    t = (t // DAY) * DAY + int(rng.uniform(0, 5) * 3600)
                mimic = rng.random() < .6
                amt = np.exp(log_med[e] + rng.normal(0, .25)) * (rng.uniform(.8, 1.5) if mimic else rng.uniform(3, 9))
                takeover = rng.random() < .35                      # uses the victim's own device
                v = noise_v()
                v[:2] += 0.35                                       # weak signal in two Vesta-like columns
                rows.append((min(t, horizon - 1), card1[e], addr1[e], round(float(amt), 2),
                             rng.choice(["C", "H", "R"], p=[.4, .35, .25]), ctype[e],
                             ring_email if rng.random() < .8 else email[e],
                             "mobile" if rng.random() < .6 else "desktop",
                             dev_ids[e][0] if takeover else ring_dev,
                             age0[e] + t / DAY, 1, *v))

    cols = ["ts", "card1", "addr1", "amount", "product", "card_type", "email_domain", "device_type",
            "cluster_key", "d1", "label", *[f"v{i}" for i in range(1, 7)]]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = df["ts"].astype("int64")
    df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(1, len(df) + 1, dtype="int64"))
    df["entity_key"] = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
    return df
