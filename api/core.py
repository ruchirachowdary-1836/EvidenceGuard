"""Shared API state: the candidate pool of the active run, default weights, helpers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import func, select

from ml.store import alerts, audit_log, experiment_runs, scores, transactions

from . import db

DAY = 86400
METHODS = ("p_only", "p_a", "p_e", "all", "full")
MODES = ("mult", "linear", "constrained")
VERDICTS = ("fraud", "legit", "escalate", "needs_info")


@dataclass
class Pool:
    run_id: str
    df: pd.DataFrame                    # one row per candidate alert
    n_day: dict[str, int]               # all test transactions per day
    fraud_day: dict[str, int]           # fraud (label) per day, evaluation only
    days: list[str]
    default_day: str
    defaults: dict[str, dict[str, float]]
    metrics: dict
    epoch: str


_cache: dict[str, Pool] = {}


def run_row(conn, run_id: str | None = None):
    rid = run_id or os.getenv("RUN_ID")
    q = select(experiment_runs)
    q = q.where(experiment_runs.c.run_id == rid) if rid else q.order_by(experiment_runs.c.started_at.desc()).limit(1)
    row = conn.execute(q).mappings().first()
    if not row:
        raise HTTPException(404, "No experiment run loaded. Run `make demo` or `python -m ml.run_experiment`.")
    return row


def day_label(epoch: str, day_idx: int) -> str:
    return (datetime.fromisoformat(epoch) + pd.Timedelta(days=int(day_idx))).date().isoformat()


def get_pool(run_id: str | None = None) -> Pool:
    with db.reader().connect() as conn:
        run = run_row(conn, run_id)
        rid = run["run_id"]
        if rid in _cache:
            return _cache[rid]
        metrics = run["metrics"] or {}
        epoch = (metrics.get("dataset") or {}).get("epoch", "2026-01-01")
        q = (select(alerts.c.alert_id, alerts.c.txn_id, scores.c.p, scores.c.a, scores.c.e, transactions.c.cluster_key,
                    transactions.c.entity_key, transactions.c.ts, transactions.c.amount, transactions.c.label)
             .select_from(alerts.join(scores, (scores.c.txn_id == alerts.c.txn_id) & (scores.c.run_id == alerts.c.run_id))
                          .join(transactions, transactions.c.txn_id == alerts.c.txn_id))
             .where(alerts.c.run_id == rid, alerts.c.method == "pool"))
        df = pd.DataFrame(conn.execute(q).mappings().all())
        if df.empty:
            raise HTTPException(404, f"Run {rid} has no alert pool")
        df["day_idx"] = df["ts"] // DAY
        df["day"] = df["day_idx"].map(lambda d: day_label(epoch, d))
        tx = pd.DataFrame(conn.execute(select(transactions.c.ts, transactions.c.label).where(transactions.c.split == "test")).all(),
                          columns=["ts", "label"])
        tx["day"] = (tx["ts"] // DAY).map(lambda d: day_label(epoch, d))
        n_day = tx.groupby("day").size().to_dict()
        fraud_day = tx.groupby("day")["label"].sum().astype(int).to_dict()
    defaults: dict[str, dict[str, float]] = {}
    for r in metrics.get("ablation", []):
        if r["mode"] == "mult" and r.get("weights", "validated") == "validated" and abs(r["budget"] - 0.05) < 1e-9:
            defaults[r["variant"]] = {"beta": r["beta"], "gamma": r["gamma"], "lam": r["lam"]}
    pool = Pool(rid, df.sort_values(["day_idx", "alert_id"]).reset_index(drop=True), n_day, fraud_day,
                sorted(df["day"].unique()), _busiest_day(df), defaults, metrics, epoch)
    _cache[rid] = pool
    return pool


def _busiest_day(df: pd.DataFrame) -> str:
    """Day with the largest candidate pool (latest wins ties); a window can end on a partial day."""
    sizes = df.groupby("day").size()
    return str(sizes[sizes == sizes.max()].index[-1])


def clear_cache() -> None:
    _cache.clear()


def audit(conn, actor: str, action: str, detail: dict) -> None:
    conn.execute(audit_log.insert().values(actor=actor, action=action, detail=detail, at=datetime.now(timezone.utc)))


def weights_for(pool: Pool, method: str, beta=None, gamma=None, lam=None) -> dict[str, float]:
    d = {"beta": 0.5, "gamma": 0.5, "lam": 0.3, **pool.defaults.get(method, {})}
    for k, v in (("beta", beta), ("gamma", gamma), ("lam", lam)):
        if v is not None:
            d[k] = float(v)
    return d


def day_arrays(pool: Pool, day: str):
    sub = pool.df[pool.df["day"] == day]
    return sub, sub["p"].to_numpy(), sub["a"].to_numpy(), sub["e"].to_numpy(), sub["cluster_key"].to_numpy()


def safe_count(conn, table) -> int:
    return conn.execute(select(func.count()).select_from(table)).scalar_one()


def check_method(method: str, mode: str = "mult") -> None:
    if method not in METHODS:
        raise HTTPException(422, f"method must be one of {METHODS}")
    if mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")


def actor_of(x_analyst: str | None) -> str:
    return (x_analyst or "analyst")[:40]


def nan_to_none(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x
