"""Loading, canonical schema, chronological splits and leakage checks.

Canonical columns: txn_id, ts (seconds), amount, product, card_type, email_domain, device_type, d1,
entity_key, cluster_key, label (+ optional v1..vN numeric extras). `label` is for evaluation only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import synth
from .config import ROOT

CANONICAL = ["txn_id", "ts", "amount", "product", "card_type", "email_domain", "device_type", "d1",
             "entity_key", "cluster_key", "label"]


def from_ieee_cis(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """IEEE-CIS Fraud Detection (Kaggle). Entity proxy = card1 + addr1 (an assumption, see README)."""
    tx = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = raw_dir / "train_identity.csv"
    if ident.exists():
        tx = tx.merge(pd.read_csv(ident, usecols=["TransactionID", "DeviceType", "DeviceInfo"]),
                      on="TransactionID", how="left")
    else:
        tx["DeviceType"], tx["DeviceInfo"] = np.nan, np.nan
    df = pd.DataFrame({
        "txn_id": tx["TransactionID"].astype("int64"),
        "ts": tx["TransactionDT"].astype("int64"),
        "amount": tx["TransactionAmt"].astype(float),
        "product": tx["ProductCD"].fillna("?"),
        "card_type": tx["card6"].fillna("?"),
        "email_domain": tx["P_emaildomain"].fillna("?"),
        "device_type": tx["DeviceType"].fillna("?"),
        "d1": tx["D1"].fillna(-1.0),
        "entity_key": tx["card1"].astype(str) + "_" + tx["addr1"].fillna(-1).astype(int).astype(str),
        "cluster_key": tx["addr1"].fillna(-1).astype(int).astype(str) + "|" + tx["DeviceInfo"].fillna("na"),
        "label": tx["isFraud"].astype("int8"),
    })
    for c in [f"V{i}" for i in range(1, 7)]:
        df[c.lower()] = tx[c].fillna(0.0).astype(float)
    return _cap(df, max_rows)


def from_paysim(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    """PaySim. Entity = origin account, cluster = destination account, ts = step * 3600."""
    path = next(iter(sorted(raw_dir.glob("PS_*.csv"))), None) or raw_dir / "paysim.csv"
    tx = pd.read_csv(path)
    df = pd.DataFrame({
        "txn_id": np.arange(1, len(tx) + 1, dtype="int64"),
        "ts": (tx["step"].astype("int64") * 3600),
        "amount": tx["amount"].astype(float),
        "product": tx["type"].astype(str),
        "card_type": "na", "email_domain": "na", "device_type": "na", "d1": -1.0,
        "entity_key": tx["nameOrig"].astype(str),
        "cluster_key": tx["nameDest"].astype(str),
        "label": tx["isFraud"].astype("int8"),
    })
    df["v1"] = tx["oldbalanceOrg"].astype(float)
    df["v2"] = tx["newbalanceOrig"].astype(float)
    return _cap(df, max_rows)


def _cap(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows and len(df) > max_rows:       # stratified, keeps all fraud and subsamples legit
        fraud, legit = df[df.label == 1], df[df.label == 0]
        keep = max(max_rows - len(fraud), 1)
        df = pd.concat([fraud, legit.sample(min(keep, len(legit)), random_state=0)])
    return df


def load_dataset(cfg: dict, seed: int | None = None) -> pd.DataFrame:
    src = cfg["data"]["source"]
    if src == "synthetic":
        s = cfg["data"]["synthetic"]
        df = synth.generate(s["n_entities"], s["n_days"], s["n_rings"], seed=cfg["seeds"][0] if seed is None else seed)
    else:
        raw = Path(cfg["data"]["raw_dir"])
        raw = raw if raw.is_absolute() else ROOT / raw
        df = from_ieee_cis(raw, cfg["data"].get("max_rows")) if src == "ieee_cis" else from_paysim(raw, cfg["data"].get("max_rows"))
    df = df.sort_values(["ts", "txn_id"], kind="mergesort").reset_index(drop=True)
    check_canonical(df)
    return df


def check_canonical(df: pd.DataFrame) -> None:
    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    if df["txn_id"].duplicated().any():
        raise ValueError("txn_id must be unique")
    if not df["ts"].is_monotonic_increasing:
        raise ValueError("rows must be sorted by time")


def chronological_folds(n: int, folds: list[list[float]]) -> list[dict[str, np.ndarray]]:
    """Index arrays for train / val / test per fold. Every fold keeps strict time order train < val < test."""
    out = []
    for tr_end, va_end, te_end in folds:
        a, b, c = int(n * tr_end), int(n * va_end), int(n * te_end)     # train [0,a) | val [a,b) | test [b,c)
        out.append({"train": np.arange(0, a), "val": np.arange(a, b), "test": np.arange(b, c)})
    return out


def assert_chronological(df: pd.DataFrame, fold: dict[str, np.ndarray]) -> None:
    """Fail loudly if any split overlaps or goes back in time."""
    ts = df["ts"].to_numpy()
    tr, va, te = ts[fold["train"]], ts[fold["val"]], ts[fold["test"]]
    if not (tr.max() <= va.min() and va.max() <= te.min()):
        raise AssertionError("splits are not chronological: train <= val <= test violated")
    ids = [set(fold[k].tolist()) for k in ("train", "val", "test")]
    if (ids[0] & ids[1]) or (ids[1] & ids[2]) or (ids[0] & ids[2]):
        raise AssertionError("splits overlap")
