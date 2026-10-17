"""One config in -> results/<run_id>/ + database rows out.

    python -m ml.run_experiment --config configs/default.yaml [--db sqlite:///...] [--no-db] [--run-id X]

Steps per chronological fold and seed: features (past-only) -> LR / XGBoost -> calibration -> anomaly score
-> candidate pool (top fraction of each day by p) -> SHAP + benign perturbations -> stability e ->
ranking variants 1-5 with weights chosen on VALIDATION only -> test metrics with bootstrap CIs.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import xgboost
import yaml
from scipy.stats import mannwhitneyu

from . import ranking
from .anomaly import AnomalyScorer
from .config import ROOT, load_config
from .data import assert_chronological, chronological_folds, load_dataset
from .features import build_features
from .metrics import bootstrap_ci, pr_auc, pr_auc_ci, pr_curve
from .models import Calibrator, fit_space, train_logreg, train_xgb
from .stability import StabilityEngine, deletion_test

DAY = 86400
CURVE_BUDGETS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15]
EPOCH = "2026-01-01"                      # synthetic calendar anchor: day 0 of the time axis


def day_label(day_idx: int) -> str:
    return (datetime.fromisoformat(EPOCH) + pd.Timedelta(days=int(day_idx))).date().isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def clean(o):
    """JSON-safe: numpy -> python, NaN -> None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---------------------------------------------------------------------------------------------
def candidate_positions(day: np.ndarray, p: np.ndarray, frac: float) -> np.ndarray:
    """Positions (into the split arrays) of the top `frac` of every day by p."""
    out = []
    for d in np.unique(day):
        pos = np.flatnonzero(day == d)
        keep = max(1, int(math.ceil(frac * len(pos))))
        out.append(pos[np.argsort(-p[pos], kind="mergesort")[:keep]])
    return np.concatenate(out)


def make_days(day, pos, p_all, a_all, e_c, y_all, clusters, cand):
    """Per-day dicts used by ranking.evaluate. `cand` are candidate positions, e_c their stability."""
    days = []
    e_full = np.full(len(day), np.nan)
    e_full[cand] = e_c
    for d in np.unique(day):
        in_day = np.flatnonzero(day == d)
        c = np.intersect1d(in_day, cand)
        days.append({"day": int(d), "pos": c, "p": p_all[c], "a": a_all[c], "e": e_full[c], "groups": clusters[c],
                     "y": y_all[c], "n_day": len(in_day), "y_total": float(y_all[in_day].sum())})
    return days


def run_fold(df, feat, fold_ix, fold, seed, cfg, log):
    t_fold = time.perf_counter()
    tr, va, te = fold["train"], fold["val"], fold["test"]
    assert_chronological(df, fold)
    space = fit_space(feat.iloc[tr])
    Xtr, Xva, Xte = (space.matrix(feat.iloc[i]) for i in (tr, va, te))
    ytr, yva, yte = (feat["label"].to_numpy()[i] for i in (tr, va, te))

    # ---- 5.1 baselines ----
    t0 = time.perf_counter()
    lr = train_logreg(Xtr, ytr, cfg["models"]["logreg_C"], seed)
    xw = train_xgb(Xtr, ytr, Xva, yva, cfg["models"]["xgb"], True, seed)
    xu = train_xgb(Xtr, ytr, Xva, yva, cfg["models"]["xgb"], False, seed)
    train_s = time.perf_counter() - t0
    models = {"logreg": lr, "xgb": xu, "xgb_weighted": xw}
    raw_val = {k: m.raw(Xva) for k, m in models.items()}
    raw_te = {k: m.raw(Xte) for k, m in models.items()}
    base_rows = {}
    for k in models:
        auc, lo, hi = pr_auc_ci(yte, raw_te[k], cfg["evaluation"]["bootstrap_n"], seed)
        base_rows[k] = {"pr_auc": auc, "ci_lo": lo, "ci_hi": hi, "val_pr_auc": pr_auc(yva, raw_val[k]),
                        "test_precision_at_0.5": float(yte[raw_te[k] >= 0.5].mean()) if (raw_te[k] >= 0.5).any() else None,
                        "test_recall_at_0.5": float((raw_te[k] >= 0.5)[yte == 1].mean()) if yte.sum() else None}
    main_key = max(("xgb", "xgb_weighted"), key=lambda k: base_rows[k]["val_pr_auc"])    # chosen on VALIDATION
    main = models[main_key]
    cal = Calibrator(cfg["models"].get("calibration", "platt")).fit(main.margin(Xva), yva)
    p_va, p_te = cal(main.margin(Xva)), cal(main.margin(Xte))
    base_rows["xgb_calibrated"] = {"pr_auc": pr_auc(yte, p_te), "ci_lo": None, "ci_hi": None, "val_pr_auc": pr_auc(yva, p_va),
                                   "brier": float(np.mean((p_te - yte) ** 2))}

    # ---- 5.2 anomaly ----
    scorer = AnomalyScorer(cfg["features"]["cold_start_n"], seed).fit(feat.iloc[tr])
    a_va, a_te = scorer.score(feat.iloc[va]), scorer.score(feat.iloc[te])

    # ---- 5.3 candidate pool + stability ----
    st_cfg = cfg["stability"]
    engine = StabilityEngine(main, cal, space.names, Xtr, st_cfg)
    day_va, day_te = feat["ts"].to_numpy()[va] // DAY, feat["ts"].to_numpy()[te] // DAY
    cand_va = candidate_positions(day_va, p_va, st_cfg["candidate_frac"])
    cand_te = candidate_positions(day_te, p_te, st_cfg["candidate_frac"])
    res_va = engine.run(Xva[cand_va], seed + 1)
    res_te = engine.run(Xte[cand_te], seed + 2)
    log(f"  fold {fold_ix}: pool val={len(cand_va)} test={len(cand_te)}  {res_te.ms_per_alert:.1f} ms/alert (SHAP + {st_cfg['M']} perturbations)")

    clusters = feat["cluster_key"].to_numpy()
    days_va = make_days(day_va, va, p_va, a_va, res_va.e, yva, clusters[va], cand_va)
    days_te = make_days(day_te, te, p_te, a_te, res_te.e, yte, clusters[te], cand_te)
    tot_va, tot_te = float(yva.sum()), float(yte.sum())

    # ---- 5.4 ranking: weights on validation, numbers on test ----
    rk = cfg["ranking"]; budgets = cfg["evaluation"]["budgets"]
    methods: dict[str, dict] = {}
    t_rank = time.perf_counter()
    for variant in ranking.VARIANTS:
        for mode in (["mult"] if variant == "p_only" else ranking.MODES):
            w = ranking.search_weights(days_va, variant, mode, rk["grid"], rk["objective_budget"], rk["risk_floor"], tot_va)
            kw = dict(variant=variant, mode=mode, beta=w["beta"], gamma=w["gamma"], lam=w["lam"], risk_floor=rk["risk_floor"])
            per_budget = {b: ranking.evaluate(days_te, b, total_fraud=tot_te, **kw) for b in budgets}
            curve = {b: ranking.evaluate(days_te, b, total_fraud=tot_te, **kw)["recall"] for b in CURVE_BUDGETS}
            methods[f"{variant}|{mode}"] = {"weights": w, "kw": kw, "test": per_budget, "curve": curve}
            if variant != "p_only":      # reference: fixed, unvalidated weights, so the table never hides the effect of a and e
                fw = {"beta": 0.5, "gamma": 0.5, "lam": 0.3}
                fkw = dict(variant=variant, mode=mode, risk_floor=rk["risk_floor"], **fw)
                methods[f"{variant}|{mode}|fixed"] = {
                    "weights": fw, "kw": fkw, "test": {b: ranking.evaluate(days_te, b, total_fraud=tot_te, **fkw) for b in budgets},
                    "curve": {b: ranking.evaluate(days_te, b, total_fraud=tot_te, **fkw)["recall"] for b in CURVE_BUDGETS}}
    rank_ms = (time.perf_counter() - t_rank) * 1000

    # ---- RQ1 + faithfulness + drift ----
    y_c = yte[cand_te]
    e_f, e_l = res_te.e[y_c == 1], res_te.e[y_c == 0]
    u = mannwhitneyu(e_f, e_l, alternative="two-sided") if len(e_f) and len(e_l) else None
    bins = np.linspace(0, 1, 21)
    stability = {
        "n_fraud": int(len(e_f)), "n_legit": int(len(e_l)),
        "mean_fraud": float(e_f.mean()) if len(e_f) else None, "mean_legit": float(e_l.mean()) if len(e_l) else None,
        "median_fraud": float(np.median(e_f)) if len(e_f) else None, "median_legit": float(np.median(e_l)) if len(e_l) else None,
        "mannwhitney_p": float(u.pvalue) if u is not None else None,
        "hist_bins": bins.round(2).tolist(),
        "hist_fraud": np.histogram(e_f, bins)[0].tolist(), "hist_legit": np.histogram(e_l, bins)[0].tolist(),
        "kept_fraction": float(res_te.kept.mean()), "ms_per_alert": res_te.ms_per_alert,
    }
    n_del = min(cfg["evaluation"]["deletion_n"], len(cand_te))
    sel = np.random.default_rng(seed).choice(len(cand_te), n_del, replace=False)
    faith = deletion_test(engine, Xte[cand_te][sel], res_te.shap[sel], res_te.p[sel], space.medians[space.names].to_numpy(np.float32), seed)
    thirds = np.array_split(np.arange(len(te)), 3)
    drift = [pr_auc(yte[t], p_te[t]) for t in thirds]

    return {
        "fold": fold_ix, "seed": seed, "main_model": main_key, "baselines": base_rows, "methods": methods,
        "stability": stability, "faithfulness": faith, "drift_pr_auc_thirds": drift,
        "efficiency": {"train_s": train_s, "ms_per_alert": res_te.ms_per_alert, "rank_ms_all_methods": rank_ms,
                       "fold_s": time.perf_counter() - t_fold},
        "curves": {"pr": {k: pr_curve(yte, raw_te[k]) for k in models} | {"xgb_calibrated": pr_curve(yte, p_te)}},
        "_keep": dict(space=space, Xte=Xte, Xva=Xva, te=te, va=va, p_te=p_te, a_te=a_te, cand_te=cand_te, res_te=res_te,
                      days_te=days_te, days_va=days_va, test_total=tot_te, yte=yte, day_te=day_te),
    }


# ---------------------------------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float('nan')


def aggregate(runs: list[dict], cfg: dict) -> dict:
    budgets, nb = cfg["evaluation"]["budgets"], cfg["evaluation"]["bootstrap_n"]
    out: dict = {"folds": len({r["fold"] for r in runs}), "seeds": len({r["seed"] for r in runs})}
    out["baselines"] = {}
    for k in runs[0]["baselines"]:
        vals = [r["baselines"][k]["pr_auc"] for r in runs]
        out["baselines"][k] = {"pr_auc_mean": float(np.nanmean(vals)), "pr_auc_std": float(np.nanstd(vals)),
                               "per_run": vals, "ci_lo": _mean([r["baselines"][k]["ci_lo"] for r in runs]),
                               "ci_hi": _mean([r["baselines"][k]["ci_hi"] for r in runs])}
    ablation = []
    for key in runs[0]["methods"]:
        variant, mode, *rest = key.split("|")
        wtype = rest[0] if rest else "validated"
        for b in budgets:
            rec = [r["methods"][key]["test"][b]["recall"] for r in runs]
            pre = [r["methods"][key]["test"][b]["precision"] for r in runs]
            red = [r["methods"][key]["test"][b]["redundancy"] for r in runs]
            tp = sum(r["methods"][key]["test"][b]["tp"] for r in runs)
            tot = sum(r["methods"][key]["test"][b]["total_fraud"] for r in runs)
            hit = np.concatenate([np.ones(int(tp)), np.zeros(int(tot - tp))])           # pooled over fold x seed
            m, lo, hi = bootstrap_ci(hit, nb, 0)
            qy = np.concatenate([np.concatenate(r["methods"][key]["test"][b]["hits"]) for r in runs])
            pm, plo, phi = bootstrap_ci(qy, nb, 1)
            ablation.append({"variant": variant, "mode": mode, "weights": wtype, "budget": b, "recall": m, "recall_lo": lo, "recall_hi": hi,
                             "recall_std_runs": float(np.std(rec)), "precision": pm, "precision_lo": plo, "precision_hi": phi,
                             "redundancy": float(np.mean(red)),
                             "beta": float(np.mean([r["methods"][key]["weights"]["beta"] for r in runs])),
                             "gamma": float(np.mean([r["methods"][key]["weights"]["gamma"] for r in runs])),
                             "lam": float(np.mean([r["methods"][key]["weights"]["lam"] for r in runs]))})
    out["ablation"] = ablation
    out["recall_vs_budget"] = {key: {str(b): float(np.mean([r["methods"][key]["curve"][b] for r in runs])) for b in CURVE_BUDGETS}
                               for key in runs[0]["methods"]}
    for k in ("faithfulness", "efficiency", "stability"):
        out[k] = {m: float(np.nanmean([np.nan if r[k][m] is None else r[k][m] for r in runs])) for m in runs[0][k] if isinstance(runs[0][k][m], (int, float))}
    out["stability"]["hist_bins"] = runs[-1]["stability"]["hist_bins"]
    out["stability"]["hist_fraud"] = np.sum([r["stability"]["hist_fraud"] for r in runs], axis=0).tolist()
    out["stability"]["hist_legit"] = np.sum([r["stability"]["hist_legit"] for r in runs], axis=0).tolist()
    out["drift_pr_auc_thirds"] = np.nanmean([r["drift_pr_auc_thirds"] for r in runs], axis=0).tolist()
    out["per_fold"] = [{"fold": r["fold"], "seed": r["seed"], "main_model": r["main_model"],
                        "xgb_pr_auc": r["baselines"][r["main_model"]]["pr_auc"], "logreg_pr_auc": r["baselines"]["logreg"]["pr_auc"],
                        "recall_p_only@5%": r["methods"]["p_only|mult"]["test"][0.05]["recall"],
                        "recall_full@5%": max(r["methods"][f"full|{m}"]["test"][0.05]["recall"] for m in ranking.MODES)} for r in runs]
    out["curves"] = {"pr": runs[-1]["curves"]["pr"], "fold": runs[-1]["fold"]}
    return out


def write_tables(outdir: Path, metrics: dict) -> None:
    pd.DataFrame(metrics["ablation"]).to_csv(outdir / "ablation.csv", index=False, float_format="%.4f")
    pd.DataFrame([{"model": k, **{a: b for a, b in v.items() if a != "per_run"}} for k, v in metrics["baselines"].items()]).to_csv(
        outdir / "baselines.csv", index=False, float_format="%.4f")
    pd.DataFrame(metrics["per_fold"]).to_csv(outdir / "per_fold.csv", index=False, float_format="%.4f")


def load_to_db(url, run_id, cfg, metrics, df, feat, last, commit, started, log):
    from sqlalchemy import insert, select

    from . import store
    eng = store.make_engine(url)
    store.md.create_all(eng)
    k = last["_keep"]; res = k["res_te"]; names = k["space"].names; te = k["te"]; cand = k["cand_te"]
    rk = cfg["ranking"]
    chosen = {key: m["kw"] for key, m in last["methods"].items()}
    all_mult, all_lin = chosen["all|mult"], chosen["all|linear"]
    p_c, a_c, e_c = k["p_te"][cand], k["a_te"][cand], res.e
    r_mult = ranking.combined_score(p_c, a_c, e_c, "mult", beta=all_mult["beta"], gamma=all_mult["gamma"])
    r_lin = ranking.combined_score(p_c, a_c, e_c, "linear", beta=all_lin["beta"], gamma=all_lin["gamma"])
    split = np.array(["train"] * len(df), dtype=object)
    split[:te[0]] = "train"
    f = last.get("_split_names")
    for name, idx in f.items():
        split[idx] = name
    cols = ["product", "card_type", "email_domain", "device_type", "d1", "hist_n", "cnt_86400", "z_amt", "hour",
            "time_since_prev", "new_device", "cluster_cnt_86400", "median_amt", "mean_amt"]
    facts = feat[cols].round(3).to_dict("records")
    with eng.begin() as conn:
        store.reset_run(conn, run_id)
        have = {r[0] for r in conn.execute(select(store.transactions.c.txn_id))}
        rows = [{"txn_id": int(df.txn_id.iat[i]), "entity_key": df.entity_key.iat[i], "cluster_key": df.cluster_key.iat[i],
                 "ts": int(df.ts.iat[i]), "amount": float(df.amount.iat[i]), "split": split[i], "label": int(df.label.iat[i]),
                 "facts": clean(facts[i])} for i in range(len(df)) if int(df.txn_id.iat[i]) not in have]
        for s in range(0, len(rows), 5000):
            conn.execute(insert(store.transactions), rows[s:s + 5000])
        conn.execute(insert(store.experiment_runs), [{"run_id": run_id, "config": clean(cfg), "git_commit": commit,
                                                      "started_at": started, "metrics": clean(metrics)}])
        tx_ids = df.txn_id.to_numpy()[te][cand]
        conn.execute(insert(store.scores), [{"run_id": run_id, "txn_id": int(tx_ids[i]), "p": float(p_c[i]), "a": float(a_c[i]),
                                              "e": float(e_c[i]), "r_linear": float(r_lin[i]), "r_mult": float(r_mult[i])}
                                             for i in range(len(cand))])
        top_n = 12
        order = np.argsort(-np.abs(res.shap), axis=1)
        Xc = k["Xte"][cand]
        ex = [{"run_id": run_id, "txn_id": int(tx_ids[i]), "feature": names[j], "shap": float(res.shap[i, j]), "rank": r + 1,
               "value": float(Xc[i, j])} for i in range(len(cand)) for r, j in enumerate(order[i, :top_n])]
        for s in range(0, len(ex), 5000):
            conn.execute(insert(store.explanations), ex[s:s + 5000])
        sr = []
        for i in range(len(cand)):
            for m in range(res.jaccard.shape[1]):
                sr.append({"run_id": run_id, "txn_id": int(tx_ids[i]), "m": m, "jaccard": float(res.jaccard[i, m]),
                           "prob_shift": float(res.prob_shift[i, m]), "kept": bool(res.kept[i, m]),
                           "topk": [names[j] for j in np.flatnonzero(res.topk[i, m])]})
        for s in range(0, len(sr), 5000):
            conn.execute(insert(store.stability_runs), sr[s:s + 5000])
        # alert pool: one stable alert_id per candidate; plus each method's default-budget queue
        days = k["days_te"]; K5 = rk["objective_budget"]
        pool = [{"run_id": run_id, "day": day_label(int(k["day_te"][c])), "method": "pool", "txn_id": int(tx_ids[i]),
                 "queue_rank": None, "budget_k": None} for i, c in enumerate(cand)]
        conn.execute(insert(store.alerts), pool)
        pos_to_txn = {int(c): int(tx_ids[i]) for i, c in enumerate(cand)}
        qrows = []
        for key, kw in chosen.items():
            if len(key.split("|")) != 2 or key.split("|")[1] != "mult":
                continue
            for d in days:
                K = min(max(1, int(math.ceil(K5 * d["n_day"]))), len(d["p"]))
                ch, _ = ranking.rank_day(d["p"], d["a"], d["e"], d["groups"], K, **kw)
                for r, i in enumerate(ch):
                    qrows.append({"run_id": run_id, "day": day_label(d["day"]), "method": key.split("|")[0],
                                  "txn_id": pos_to_txn[int(d["pos"][i])], "queue_rank": r + 1, "budget_k": K})
        for s in range(0, len(qrows), 5000):
            conn.execute(insert(store.alerts), qrows[s:s + 5000])
        conn.execute(insert(store.audit_log), [{"actor": "pipeline", "action": "load_run", "detail": {"run_id": run_id, "alerts": len(pool)}}])
    log(f"  database: {len(rows)} new transactions, {len(cand)} scored candidates, {len(ex)} explanation rows -> {url or store.DEFAULT_URL}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--db", default=None)
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    started = datetime.now(timezone.utc)
    run_id = a.run_id or f"{cfg['name']}-{started:%Y%m%d-%H%M%S}"
    outdir = ROOT / "results" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "run.log", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"run {run_id}  config={a.config}  commit={git_commit()}")
    t0 = time.perf_counter()
    df = load_dataset(cfg)
    feat = build_features(df, cfg["features"]["windows_s"], cfg["features"]["history_cap"])
    log(f"data: {len(df):,} rows, {int(df.label.sum()):,} fraud ({df.label.mean():.2%}), {df.ts.max() / DAY:.0f} days, "
        f"{df.entity_key.nunique():,} entities  [{cfg['data']['source']}]")
    folds = chronological_folds(len(df), cfg["splits"]["folds"])
    runs = []
    for fi, fold in enumerate(folds, 1):
        for seed in cfg["seeds"]:
            r = run_fold(df, feat, fi, fold, seed, cfg, log)
            r["_split_names"] = fold
            runs.append(r)
            b = r["baselines"]
            log(f"  fold {fi} seed {seed}: PR-AUC logreg={b['logreg']['pr_auc']:.3f} xgb={b['xgb']['pr_auc']:.3f} "
                f"xgb_w={b['xgb_weighted']['pr_auc']:.3f}  main={r['main_model']}  e fraud/legit="
                f"{r['stability']['mean_fraud']:.2f}/{r['stability']['mean_legit']:.2f}")
    metrics = aggregate(runs, cfg)
    metrics.update({"run_id": run_id, "dataset": {"source": cfg["data"]["source"], "rows": len(df), "fraud_rate": float(df.label.mean()),
                                                  "entities": int(df.entity_key.nunique()), "epoch": EPOCH,
                                                  "caveat": ("Synthetic data: demonstrates the pipeline, not real-world fraud detection."
                                                             if cfg["data"]["source"] == "synthetic" else
                                                             "Public benchmark data, not Indian UPI traffic.")},
                    "model_version": f"xgb-{xgboost.__version__}+{run_id}", "runtime_s": time.perf_counter() - t0})
    metrics = clean(metrics)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (outdir / "meta.json").write_text(json.dumps({"git_commit": git_commit(), "python": platform.python_version(), "started_at": started.isoformat(),
                                                  "libs": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                                                           "xgboost": xgboost.__version__, "shap": shap.__version__, "scipy": scipy.__version__}}, indent=2))
    write_tables(outdir, metrics)
    if not a.no_db:
        load_to_db(a.db, run_id, cfg, metrics, df, feat, runs[-1], git_commit(), started, log)
    log(f"done in {time.perf_counter() - t0:.0f}s -> results/{run_id}/")
    return run_id


if __name__ == "__main__":
    main()
