# EvidenceGuard

**Explanation-aware, budgeted alert ranking for card fraud, with an analyst console.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-SHAP-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-SQLAlchemy-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB?logo=react&logoColor=black)
![Tests](https://img.shields.io/badge/tests-41%20python%20%7C%2010%20web-22c55e)

> **Dataset caveat.** EvidenceGuard is a research prototype. It runs on a *synthetic* simulator (bundled), and has
> loaders for IEEE-CIS and PaySim. **None of these is real Indian UPI traffic and nothing here claims UPI
> effectiveness.** The dashboard footer repeats this on every page.

![Budgeted queue](docs/queue.jpg)

## The idea

A fraud model gives every transaction a probability *p*. Analysts can only review *K* alerts a day, and a
ranked list of near-duplicates from one stolen device wastes that budget. EvidenceGuard tests three additions to
plain *p*-ranking and reports honestly whether each one helps:

| Component | Meaning |
|-----------|---------|
| **a** - personalised anomaly | how unusual this transaction is for *this customer* (robust z of log-amount, hour, product; IsolationForest cold-start) |
| **e** - explanation stability | mean Jaccard of the top-5 SHAP features across 20 *benign* perturbations (amount jitter, time shift, resampled non-essential fields; perturbations that move *p* by more than delta are discarded) |
| **queue** | greedy top-K where each extra alert from the same cluster (device group) is discounted by (1 - lambda) |

Combined score: `R = p (1 + beta a)(1 - gamma + gamma e)` (multiplicative: cannot rescue a low *p*), with the plan's
linear baseline `alpha p + beta a + gamma e` and a *constrained* variant for comparison. Weights are chosen on
**validation only** (objective: recall at 5% review budget); numbers are reported on **test** windows.

Five variants are evaluated: **1** p only, **2** p + a, **3** p + e, **4** all components, **5** full method with the redundancy-aware queue.

## What is in the box

- **ML pipeline** (`ml/`): leakage-safe past-only features, chronological 3-fold protocol, Logistic Regression and
  XGBoost (weighted vs unweighted), Platt/isotonic calibration, anomaly score, SHAP + perturbation stability,
  faithfulness deletion test, ranking variants, weight search, bootstrap CIs, drift and latency measurements.
- **One config, one command**: `python -m ml.run_experiment --config configs/default.yaml` writes
  `results/<run_id>/` (metrics.json, ablation.csv, baselines.csv, config, git commit, library versions) and the database rows.
- **Database**: the plan's PostgreSQL schema (SQLAlchemy; SQLite for local runs) with parameterised queries, a
  read-only role for read endpoints, and an audit log.
- **API** (FastAPI): ranked queue, evidence, decisions, simulator, experiments, audit trail + CSV, a server-sent-events feed.
- **Analyst console** (React, TypeScript, Tailwind, Recharts) - dark and dense, animated, keyboard-driven:

  | Page | Highlights |
  |------|-----------|
  | **Queue** | budget slider K, 5-way method switch with sliding thumb, lambda/beta/gamma sliders, **rank movement vs p-only** per row, cluster tags with live duplicate counts, **live candidate feed** (SSE replay), `j`/`k`/`Enter` keyboard navigation |
  | **Evidence** | animated risk gauge, SHAP bar chart, **perturbation heatmap** (which features enter/leave the top-5 under each perturbation; discarded perturbations hatched), customer baseline vs this transaction, peers in the cluster, decision form |
  | **Models** | PR-AUC with CIs, PR curves, ablation table with 95% CIs (validated vs fixed weights, three score forms), recall-vs-budget curves, stability distribution fraud vs legit (Mann-Whitney), faithfulness, drift, latency - with an automatic "reliable improvement or not" note |
  | **Simulator** | live recall / precision / redundancy against the p-only baseline as you move K, alpha, beta, gamma, lambda |
  | **Audit** | who viewed or decided what and when, filter, auto-refresh, CSV export |

  Risk is shown by colour *and* a numeric bar; stability has its own glyph + word + bar (never colour alone). Command palette
  (`Ctrl K`), loading skeletons and empty/error states everywhere, reduced-motion support, responsive down to tablet width.

![Evidence view](docs/evidence.jpg)

## Quick start

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
# the repo ships a small demo database (data/demo.db, ~7.8k synthetic transactions)
DATABASE_URL=sqlite:///data/demo.db uvicorn api.main:app --port 8000        # API  (Windows: set DATABASE_URL=... first)
cd web && npm install && npm run dev                                         # dashboard on http://localhost:5173
```

Or build once and let the API serve the dashboard: `cd web && npm install && npm run build`, then open `http://localhost:8000`.

With `make` (Linux/macOS/WSL): `make demo` (rebuild the demo DB), `make api`, `make web`, `make ablate` (full protocol), `make test`, `make test-web`.

### Docker (PostgreSQL + API + web)

```bash
docker compose up --build        # dashboard http://localhost:8080 (API :8000, PostgreSQL volume)
```

The API image loads the shipped demo database into PostgreSQL on first start (`ml.copy_db --if-empty`) and uses a
read-only role for read endpoints. *The compose setup is written but was not run in the environment this was built in
(no Docker available) - treat it as untested until you run it.*

## Reproducing results

```bash
python -m ml.run_experiment --config configs/default.yaml     # 3 folds x 1 seed
python -m ml.run_experiment --config configs/full.yaml        # 3 folds x 5 seeds, 2000 bootstrap resamples
```

Everything is seeded; the same config and seed give identical features and scores (tested). Real data: put the Kaggle
IEEE-CIS files (`train_transaction.csv`, `train_identity.csv`) or PaySim (`PS_*.csv`) in `data/raw/` and set
`data.source: ieee_cis` or `paysim`. **The loaders are unit-tested on tiny fixtures only; they have not been run on the real files.**

### Leakage rules (enforced by tests)

- A feature for transaction *t* uses only rows with time **strictly earlier** than *t* (ties excluded). Test: recompute
  on a dataset cut at time T, and with a changed future row - earlier rows must be identical.
- Target encoding, imputation and scaling are fitted on the training window; calibration on validation.
- No label or outcome field is a model input; labels are evaluation-only (the API hides them unless `reveal=true`).
- Splits are chronological and validated (`assert_chronological` fails loudly); a shuffled-time test confirms performance drops when time order is destroyed.
- Entity proxy: `card1 + addr1` (an assumption, as in the plan). Because it contains the address, "new address" cannot occur; flags are new-device and new-email.

## Demo results (synthetic data - a pipeline demonstration, not a finding)

From `configs/demo.yaml` (7,789 transactions, 3.7% fraud, 3 chronological folds, 1 seed):

| Model (test PR-AUC) | logreg | xgboost | xgboost weighted | calibrated main |
|---|---|---|---|---|
| mean over folds | 0.922 | 0.933 | 0.947 | 0.946 |

Ranking at a 5% daily review budget (recall of all fraud, 95% bootstrap CI):

| Variant | validated weights | recall@5% |
|---|---|---|
| 1. p only | - | 58.8% (51-65) |
| 2-5. with a, e, queue | **beta = gamma = lambda = 0 selected on validation** | 58.8% (identical to p only) |
| 5. full, *fixed* weights 0.5 / 0.5 / 0.3 | not validated | 54.2% (47-61); redundancy 51% -> 41% |

What this honestly says: on this simulator, anomaly and stability did **not** improve recall on validation, so the
validation search kept them at zero and the method reduces to p-only. Explanation stability was only weakly different
for fraud vs legitimate alerts (mean 0.71 vs 0.66, Mann-Whitney p = 0.054). The deletion test shows SHAP's top-5 features
are more influential than random ones (mean p drop 0.22 vs 0.06). The redundancy-aware queue cuts duplicate alerts a lot
at small budgets (1% budget: 47% -> 4% redundancy with unchanged recall) but costs recall at 5%. That is the plan's
"valid negative result" outcome; the dashboard says so automatically instead of hiding it.

![Models page](docs/models.jpg)

## Project layout

```
configs/    YAML: data, splits, features, models, stability, ranking grids, evaluation (default, full, demo)
ml/         data.py features.py models.py anomaly.py stability.py ranking.py metrics.py run_experiment.py
            store.py (schema)  synth.py (simulator)  copy_db.py  make_data.py  config.py
api/        main.py schemas.py db.py core.py routers/{alerts,simulate,experiments,audit,stream}.py
web/        React + TypeScript + Tailwind dashboard, vitest tests (mocked API)
tests/      leakage, stability, ranking, metrics, data loaders, database copy, API (real tiny end-to-end run)
db/init.sql read-only role; Dockerfile.api / Dockerfile.web / docker-compose.yml; Makefile; results/ (versioned run output)
```

## API

Interactive docs at `/docs`.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/alerts?day=&method=&k=&lam=&mode=&beta=&gamma=` | ranked, budgeted queue |
| GET | `/api/alerts/{id}` | evidence: scores, SHAP, perturbation runs, customer baseline, cluster peers |
| POST | `/api/alerts/{id}/decision` | analyst verdict (writes `decisions` + `audit_log`) |
| POST | `/api/simulate` | re-rank with chosen weights, K, lambda; recall/precision/redundancy vs p-only |
| GET | `/api/experiments`, `/api/experiments/{run}/curves` | registry, PR curves, recall vs budget, ablation |
| GET | `/api/audit`, `/api/audit/export.csv` | paged audit trail (CSV is formula-injection safe) |
| GET | `/api/stream` | SSE replay of held-out candidate alerts (not live traffic) |
| GET | `/api/health`, `/api/meta` | liveness; run, model version, dataset caveat |

Security basics: parameterised SQL (SQLAlchemy), CORS limited to the web origin, no secrets in git, read-only role for
read endpoints in Docker. The prototype has **no user authentication**: the analyst name is self-declared and recorded
in the audit log - add real auth before any shared deployment.

## Testing

```bash
python -m pytest -q          # 41 tests: leakage, determinism, stability, ranking, metrics, loaders, API, DB copy
cd web && npm test           # 10 tests: every page against a mocked API, queue keyboard use, decision POST, error state
```

## Limitations

- Results on synthetic data demonstrate the pipeline only. The real-data loaders are untested on the real files.
- The perturbation set is one specific definition of "benign"; sensitivity to the 2% vs 5% amount tolerance and to the
  delta threshold is configurable but not yet swept.
- Stability is computed for the top 20% of each day by p (the candidate pool), so every method re-ranks the same pool;
  recall is still measured against *all* fraud in the window.
- Entity proxies are assumptions; sensitivity to alternatives is not yet run.
- Docker Compose is untested here; no auth in the API.

## License

MIT
