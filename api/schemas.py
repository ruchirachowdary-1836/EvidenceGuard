from __future__ import annotations

from pydantic import BaseModel, Field


class AlertOut(BaseModel):
    alert_id: int
    txn_id: int
    queue_rank: int
    p: float
    a: float
    e: float
    score: float
    top_features: list[tuple[str, float]]
    cluster_key: str
    cluster_size: int                 # alerts of this cluster inside the current queue
    amount: float
    ts: int
    verdict: str | None = None


class QueueOut(BaseModel):
    run_id: str
    day: str
    method: str
    k: int
    lam: float
    weights: dict[str, float]
    n_pool: int
    n_day: int
    redundancy: float
    alerts: list[AlertOut]


class SimulateIn(BaseModel):
    method: str = "all"
    mode: str = "mult"
    k: int = Field(100, ge=1, le=5000, description="review capacity per day")
    lam: float = Field(0.3, ge=0, le=0.95)
    alpha: float = 1.0
    beta: float = Field(0.5, ge=0, le=5)
    gamma: float = Field(0.5, ge=0, le=1)
    day: str | None = None
    run_id: str | None = None


class DecisionIn(BaseModel):
    analyst: str = Field("analyst", min_length=1, max_length=40)
    verdict: str
    note: str = Field("", max_length=1000)
