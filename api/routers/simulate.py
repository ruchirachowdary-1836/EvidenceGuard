from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from api import core
from api.schemas import SimulateIn
from ml import ranking

router = APIRouter(prefix="/api", tags=["simulator"])


def _days(pool, only: str | None):
    out = []
    for day in ([only] if only else pool.days):
        sub, p, a, e, groups = core.day_arrays(pool, day)
        out.append({"day": day, "p": p, "a": a, "e": e, "groups": groups, "y": sub["label"].to_numpy(), "n_day": pool.n_day.get(day, len(sub)),
                    "y_total": float(pool.fraud_day.get(day, 0))})
    return out


@router.post("/simulate")
def simulate(body: SimulateIn):
    core.check_method(body.method, body.mode)
    pool = core.get_pool(body.run_id)
    if body.day and body.day not in pool.days:
        raise HTTPException(404, "Unknown day")
    days = _days(pool, body.day)
    total = float(sum(d["y_total"] for d in days))
    kw = dict(variant=body.method, mode=body.mode, beta=body.beta, gamma=body.gamma, lam=body.lam, alpha=body.alpha)

    def run(k, **over):
        return ranking.evaluate(days, k_abs=k, total_fraud=total, **{**kw, **over})

    cur = run(body.k)
    base = run(body.k, variant="p_only", beta=0, gamma=0, lam=0)
    per_day = []
    for d in days:
        K = min(body.k, len(d["p"]))
        ch, _ = ranking.rank_day(d["p"], d["a"], d["e"], d["groups"], K, **kw)
        per_day.append({"day": d["day"], "k": K, "tp": float(d["y"][ch].sum()), "fraud": d["y_total"], "n_day": d["n_day"]})
    grid = sorted({max(1, int(round(body.k * f))) for f in (0.15, 0.3, 0.5, 0.75, 1, 1.25, 1.5, 2, 3)})
    curve = [{"k": k, "recall": run(k)["recall"], "baseline": run(k, variant="p_only", beta=0, gamma=0, lam=0)["recall"]} for k in grid]
    return {
        "run_id": pool.run_id, "k": body.k, "days": len(days), "total_fraud": total,
        "recall": cur["recall"], "precision": cur["precision"], "redundancy": cur["redundancy"], "fraud_caught": cur["tp"], "queued": cur["queued"],
        "baseline": {"recall": base["recall"], "precision": base["precision"], "redundancy": base["redundancy"], "fraud_caught": base["tp"]},
        "delta_recall": cur["recall"] - base["recall"], "per_day": per_day, "curve": curve,
        "note": "Review capacity K is per day; recall counts all fraud in the window, including fraud outside the stability-scored pool.",
    }
