"""Relational store (PostgreSQL in production, SQLite for local runs) - schema from the build plan.

Extensions over the plan's DDL: transactions.facts (JSON of display fields), stability_runs.topk (which
features were in the top-k after each perturbation, for the heatmap) and alerts.fold/pool rows.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import (JSON, BigInteger, Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Integer,
                        MetaData, SmallInteger, String, Table, Text, create_engine, event, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = f"sqlite:///{(ROOT / 'data' / 'evidenceguard.db').as_posix()}"
J = JSON().with_variant(JSONB(), "postgresql")
BIG = BigInteger().with_variant(Integer(), "sqlite")           # sqlite only auto-increments INTEGER keys

md = MetaData()

transactions = Table(
    "transactions", md,
    Column("txn_id", BigInteger, primary_key=True, autoincrement=False), Column("entity_key", String, index=True),
    Column("cluster_key", String, index=True), Column("ts", BigInteger, nullable=False, index=True),
    Column("amount", Float), Column("split", String), Column("label", SmallInteger),   # label: evaluation only
    Column("facts", J), CheckConstraint("split IN ('train','val','test')", name="ck_split"))

scores = Table(
    "scores", md, Column("run_id", String, primary_key=True),
    Column("txn_id", BigInteger, ForeignKey("transactions.txn_id"), primary_key=True),
    Column("p", Float), Column("a", Float), Column("e", Float), Column("r_linear", Float), Column("r_mult", Float))

explanations = Table(
    "explanations", md, Column("run_id", String, primary_key=True), Column("txn_id", BigInteger, primary_key=True),
    Column("feature", String, primary_key=True), Column("shap", Float), Column("rank", SmallInteger),
    Column("value", Float))

stability_runs = Table(
    "stability_runs", md, Column("id", BIG, primary_key=True, autoincrement=True), Column("run_id", String, index=True),
    Column("txn_id", BigInteger, index=True), Column("m", SmallInteger), Column("jaccard", Float),
    Column("prob_shift", Float), Column("kept", Boolean), Column("topk", J))

alerts = Table(
    "alerts", md, Column("alert_id", BIG, primary_key=True, autoincrement=True), Column("run_id", String, index=True),
    Column("day", String, index=True), Column("method", String, index=True),
    Column("txn_id", BigInteger, index=True), Column("queue_rank", Integer), Column("budget_k", Integer))

decisions = Table(
    "decisions", md, Column("id", BIG, primary_key=True, autoincrement=True),
    Column("alert_id", BigInteger, ForeignKey("alerts.alert_id"), index=True), Column("analyst", String),
    Column("verdict", String), Column("note", Text), Column("decided_at", DateTime(timezone=True), server_default=func.now()))

experiment_runs = Table(
    "experiment_runs", md, Column("run_id", String, primary_key=True), Column("config", J), Column("git_commit", String),
    Column("started_at", DateTime(timezone=True)), Column("metrics", J))

audit_log = Table(
    "audit_log", md, Column("id", BIG, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), server_default=func.now(), index=True), Column("actor", String),
    Column("action", String), Column("detail", J))


def make_engine(url: str | None = None) -> Engine:
    url = url or os.getenv("DATABASE_URL", DEFAULT_URL)
    if url.startswith("sqlite"):
        Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        eng = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(eng, "connect")
        def _fk(dbapi, _):                                          # pragma: no cover - driver hook
            dbapi.execute("PRAGMA foreign_keys=ON")
        return eng
    return create_engine(url, pool_pre_ping=True)


def reset_run(conn, run_id: str) -> None:
    """Idempotent re-load: remove one run's rows (children first)."""
    for t in (stability_runs, explanations, scores):
        conn.execute(t.delete().where(t.c.run_id == run_id))
    ids = [r[0] for r in conn.execute(alerts.select().with_only_columns(alerts.c.alert_id).where(alerts.c.run_id == run_id))]
    if ids:
        conn.execute(decisions.delete().where(decisions.c.alert_id.in_(ids)))
    conn.execute(alerts.delete().where(alerts.c.run_id == run_id))
    conn.execute(experiment_runs.delete().where(experiment_runs.c.run_id == run_id))
