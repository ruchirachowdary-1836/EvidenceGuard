from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from api import core, db
from ml.store import experiment_runs

router = APIRouter(prefix="/api", tags=["experiments"])


def _headline(m: dict) -> dict:
    ab = [r for r in m.get("ablation", []) if abs(r["budget"] - 0.05) < 1e-9 and r.get("weights", "validated") == "validated"]
    base = next((r for r in ab if r["variant"] == "p_only"), None)
    best = max((r for r in ab if r["variant"] != "p_only"), key=lambda r: r["recall"], default=None)
    return {"pr_auc_xgb": (m.get("baselines", {}).get("xgb_calibrated") or {}).get("pr_auc_mean"),
            "pr_auc_logreg": (m.get("baselines", {}).get("logreg") or {}).get("pr_auc_mean"),
            "recall5_p_only": base and base["recall"], "recall5_best": best and best["recall"],
            "best_method": best and f"{best['variant']}|{best['mode']}"}


@router.get("/experiments")
def experiments():
    with db.reader().connect() as conn:
        rows = conn.execute(select(experiment_runs).order_by(experiment_runs.c.started_at.desc())).mappings().all()
    return [{"run_id": r["run_id"], "name": (r["config"] or {}).get("name"), "started_at": str(r["started_at"]), "git_commit": r["git_commit"],
             "dataset": (r["metrics"] or {}).get("dataset"), "folds": (r["metrics"] or {}).get("folds"), "seeds": (r["metrics"] or {}).get("seeds"),
             "headline": _headline(r["metrics"] or {}), "config": r["config"]} for r in rows]


@router.get("/experiments/{run_id}/curves")
def curves(run_id: str):
    with db.reader().connect() as conn:
        try:
            run = core.run_row(conn, run_id)
        except HTTPException:
            raise HTTPException(404, "Unknown run")
    m = run["metrics"] or {}
    keys = ("ablation", "baselines", "stability", "faithfulness", "efficiency", "drift_pr_auc_thirds", "per_fold", "recall_vs_budget", "curves", "dataset")
    return {"run_id": run_id, **{k: m.get(k) for k in keys}}
