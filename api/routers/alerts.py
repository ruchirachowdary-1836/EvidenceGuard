from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Query
from sqlalchemy import select

from api import core, db
from api.schemas import AlertOut, DecisionIn, QueueOut
from ml import ranking
from ml.store import alerts, decisions, explanations, scores, stability_runs, transactions

router = APIRouter(prefix="/api", tags=["alerts"])


def _verdicts(conn, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    rows = conn.execute(select(decisions.c.alert_id, decisions.c.verdict).where(decisions.c.alert_id.in_(ids)).order_by(decisions.c.id)).all()
    return {a: v for a, v in rows}


@router.get("/alerts", response_model=QueueOut)
def list_alerts(day: str | None = None, method: str = "full", k: Annotated[int | None, Query(ge=1, le=5000)] = None,
                lam: Annotated[float | None, Query(ge=0, le=0.95)] = None, mode: str = "mult",
                beta: float | None = None, gamma: float | None = None, run_id: str | None = None):
    core.check_method(method, mode)
    pool = core.get_pool(run_id)
    day = day or pool.default_day
    if day not in pool.days:
        raise HTTPException(404, f"No alerts for {day}. Available: {pool.days[0]} .. {pool.days[-1]}")
    sub, p, a, e, groups = core.day_arrays(pool, day)
    n_day = pool.n_day.get(day, len(sub))
    K = min(k or max(1, math.ceil(0.05 * n_day)), len(sub))
    w = core.weights_for(pool, method, beta, gamma, lam)
    chosen, score = ranking.rank_day(p, a, e, groups, K, variant=method, mode=mode, beta=w["beta"], gamma=w["gamma"], lam=w["lam"])
    ids = [int(sub["txn_id"].iat[i]) for i in chosen]
    top: dict[int, list] = {}
    with db.reader().connect() as conn:
        for t, f, s, r in conn.execute(select(explanations.c.txn_id, explanations.c.feature, explanations.c.shap, explanations.c.rank)
                                       .where(explanations.c.run_id == pool.run_id, explanations.c.txn_id.in_(ids), explanations.c.rank <= 3)
                                       .order_by(explanations.c.rank)):
            top.setdefault(t, []).append((f, float(s)))
        verdict = _verdicts(conn, [int(sub["alert_id"].iat[i]) for i in chosen])
    cluster_n: dict[str, int] = {}
    for i in chosen:
        cluster_n[groups[i]] = cluster_n.get(groups[i], 0) + 1
    out = [AlertOut(alert_id=int(sub["alert_id"].iat[i]), txn_id=int(sub["txn_id"].iat[i]), queue_rank=r + 1, p=float(p[i]), a=float(a[i]),
                    e=float(e[i]), score=float(score[i]), top_features=top.get(int(sub["txn_id"].iat[i]), []),
                    cluster_key=str(groups[i]), cluster_size=cluster_n[groups[i]], amount=float(sub["amount"].iat[i]),
                    ts=int(sub["ts"].iat[i]), verdict=verdict.get(int(sub["alert_id"].iat[i]))) for r, i in enumerate(chosen)]
    return QueueOut(run_id=pool.run_id, day=day, method=method, k=K, lam=w["lam"] if method == "full" else 0.0,
                    weights={"alpha": 1.0, **w}, n_pool=len(sub), n_day=n_day, redundancy=ranking.redundancy(groups, chosen), alerts=out)


@router.get("/alerts/{alert_id}")
def alert_detail(alert_id: int, reveal: bool = False, run_id: str | None = None,
                 x_analyst: Annotated[str | None, Header()] = None):
    pool = core.get_pool(run_id)
    actor = core.actor_of(x_analyst)
    with db.reader().connect() as conn:
        al = conn.execute(select(alerts).where(alerts.c.alert_id == alert_id, alerts.c.run_id == pool.run_id, alerts.c.method == "pool")).mappings().first()
        if not al:
            raise HTTPException(404, "Alert not found")
        tx = conn.execute(select(transactions).where(transactions.c.txn_id == al["txn_id"])).mappings().one()
        sc = conn.execute(select(scores).where(scores.c.run_id == pool.run_id, scores.c.txn_id == al["txn_id"])).mappings().one()
        shap_rows = conn.execute(select(explanations).where(explanations.c.run_id == pool.run_id, explanations.c.txn_id == al["txn_id"])
                                 .order_by(explanations.c.rank)).mappings().all()
        runs = conn.execute(select(stability_runs).where(stability_runs.c.run_id == pool.run_id, stability_runs.c.txn_id == al["txn_id"])
                            .order_by(stability_runs.c.m)).mappings().all()
        hist = conn.execute(select(transactions.c.ts, transactions.c.amount).where(transactions.c.entity_key == tx["entity_key"], transactions.c.ts < tx["ts"])
                            .order_by(transactions.c.ts.desc()).limit(60)).all()
        peers = conn.execute(select(alerts.c.alert_id, transactions.c.txn_id, transactions.c.ts, transactions.c.amount, transactions.c.entity_key,
                                    scores.c.p, scores.c.e, transactions.c.label)
                             .select_from(transactions.outerjoin(alerts, (alerts.c.txn_id == transactions.c.txn_id) & (alerts.c.method == "pool") & (alerts.c.run_id == pool.run_id))
                                          .outerjoin(scores, (scores.c.txn_id == transactions.c.txn_id) & (scores.c.run_id == pool.run_id)))
                             .where(transactions.c.cluster_key == tx["cluster_key"], transactions.c.txn_id != tx["txn_id"],
                                    transactions.c.ts.between(tx["ts"] - 2 * core.DAY, tx["ts"] + 2 * core.DAY))
                             .order_by(transactions.c.ts).limit(25)).mappings().all()
        verdict_rows = conn.execute(select(decisions).where(decisions.c.alert_id == alert_id).order_by(decisions.c.id.desc())).mappings().all()
    amounts = np.array([h[1] for h in hist], float)
    med = float(np.median(amounts)) if len(amounts) else None
    mad = float(np.median(np.abs(amounts - med))) if len(amounts) else None
    hours = np.histogram([(h[0] % core.DAY) // 3600 for h in hist], bins=np.arange(25))[0].tolist() if len(hist) else [0] * 24
    with db.writer().begin() as conn:
        core.audit(conn, actor, "view_alert", {"alert_id": alert_id, "txn_id": int(al["txn_id"])})
    return {
        "alert_id": alert_id, "run_id": pool.run_id, "day": pool.df.loc[pool.df.alert_id == alert_id, "day"].iat[0],
        "transaction": {"txn_id": int(tx["txn_id"]), "entity_key": tx["entity_key"], "cluster_key": tx["cluster_key"], "ts": int(tx["ts"]),
                        "amount": float(tx["amount"]), "facts": tx["facts"], "hour": float((tx["ts"] % core.DAY) / 3600),
                        "label": int(tx["label"]) if reveal else None},
        "scores": {"p": sc["p"], "a": sc["a"], "e": sc["e"], "r_linear": sc["r_linear"], "r_mult": sc["r_mult"]},
        "shap": [{"feature": r["feature"], "shap": r["shap"], "rank": r["rank"], "value": r["value"]} for r in shap_rows],
        "stability": [{"m": r["m"], "jaccard": r["jaccard"], "prob_shift": r["prob_shift"], "kept": bool(r["kept"]), "topk": r["topk"]} for r in runs],
        "baseline": {"history": [{"ts": int(t), "amount": float(a)} for t, a in reversed(hist)], "median": med, "mad": mad, "hours": hours,
                     "n": len(hist)},
        "peers": [{"alert_id": r["alert_id"], "txn_id": int(r["txn_id"]), "ts": int(r["ts"]), "amount": float(r["amount"]), "entity_key": r["entity_key"],
                   "p": r["p"], "e": r["e"], "label": int(r["label"]) if reveal else None} for r in peers],
        "decisions": [{"analyst": r["analyst"], "verdict": r["verdict"], "note": r["note"], "decided_at": str(r["decided_at"])} for r in verdict_rows],
    }


@router.post("/alerts/{alert_id}/decision", status_code=201)
def decide(alert_id: int, body: DecisionIn):
    if body.verdict not in core.VERDICTS:
        raise HTTPException(422, f"verdict must be one of {core.VERDICTS}")
    with db.writer().begin() as conn:
        al = conn.execute(select(alerts.c.alert_id, alerts.c.txn_id).where(alerts.c.alert_id == alert_id)).first()
        if not al:
            raise HTTPException(404, "Alert not found")
        conn.execute(decisions.insert().values(alert_id=alert_id, analyst=body.analyst, verdict=body.verdict, note=body.note))
        core.audit(conn, body.analyst, "decision", {"alert_id": alert_id, "txn_id": int(al[1]), "verdict": body.verdict, "note": body.note})
    return {"alert_id": alert_id, "verdict": body.verdict, "analyst": body.analyst}
