import type { AlertRow, Curves, Detail, ExperimentRow, Meta, Queue, SimResult } from "../api";

export const meta: Meta = {
  run_id: "test-run", model_version: "xgb-test", caveat: "Research prototype on synthetic data. Not real Indian UPI traffic; no claim of UPI effectiveness.",
  days: ["2026-02-01", "2026-02-02"], default_day: "2026-02-02", methods: ["p_only", "p_a", "p_e", "all", "full"], modes: ["mult", "linear", "constrained"],
  defaults: { full: { beta: 0.5, gamma: 0.25, lam: 0.2 } }, pool_size: 80, n_day: { "2026-02-01": 200, "2026-02-02": 220 }, default_budget: 0.05,
  dataset: { source: "synthetic", rows: 5000, fraud_rate: 0.03, entities: 100, caveat: "Synthetic data: demonstrates the pipeline, not real-world fraud detection." },
};

const row = (i: number, over: Partial<AlertRow> = {}): AlertRow => ({
  alert_id: 100 + i, txn_id: 5000 + i, queue_rank: i, p: 0.9 - i * 0.1, a: 0.6, e: 0.3 + i * 0.2, score: 0.8 - i * 0.1,
  top_features: [["z_amt", 1.2], ["new_device", 0.8]], cluster_key: i < 3 ? "ringdev1" : `dev${i}`, cluster_size: i < 3 ? 2 : 1, amount: 100 + i,
  ts: 86400 * 3 + i * 3600, verdict: null, ...over,
});
export const queue: Queue = {
  run_id: "test-run", day: "2026-02-02", method: "full", k: 3, lam: 0.2, weights: { alpha: 1, beta: 0.5, gamma: 0.25, lam: 0.2 },
  n_pool: 44, n_day: 220, redundancy: 0.67, alerts: [row(1), row(2), row(3, { verdict: "fraud" })],
};
export const baseQueue: Queue = { ...queue, method: "p_only", alerts: [row(1), row(3), row(2)] };

export const detail: Detail = {
  alert_id: 101, run_id: "test-run", day: "2026-02-02",
  transaction: { txn_id: 5001, entity_key: "1234_55", cluster_key: "ringdev1", ts: 86400 * 3 + 3600, amount: 101, hour: 1, label: null, facts: { product: "C", hist_n: 12, z_amt: 3.2 } },
  scores: { p: 0.8, a: 0.6, e: 0.5, r_linear: 1.2, r_mult: 0.7 },
  shap: [{ feature: "z_amt", shap: 1.4, rank: 1, value: 3.2 }, { feature: "new_device", shap: 0.9, rank: 2, value: 1 }, { feature: "hour_dev", shap: -0.4, rank: 3, value: 5 }],
  stability: [0, 1, 2, 3].map((m) => ({ m, jaccard: 0.6, prob_shift: 0.01, kept: m !== 3, topk: ["z_amt", "new_device", m % 2 ? "hour_dev" : "v1"] })),
  baseline: { history: [{ ts: 86400, amount: 40 }, { ts: 86400 * 2, amount: 55 }], median: 47, mad: 7, hours: new Array(24).fill(0).map((_, i) => (i === 14 ? 5 : 0)), n: 2 },
  peers: [{ alert_id: 102, txn_id: 5002, ts: 86400 * 3, amount: 80, entity_key: "999_1", p: 0.7, e: 0.4, label: null }], decisions: [],
};

export const sim: SimResult = {
  run_id: "test-run", k: 30, days: 2, total_fraud: 20, recall: 0.55, precision: 0.3, redundancy: 0.2, fraud_caught: 11, queued: 40, delta_recall: 0.05,
  baseline: { recall: 0.5, precision: 0.28, redundancy: 0.4, fraud_caught: 10 }, per_day: [{ day: "2026-02-01", k: 30, tp: 5, fraud: 10, n_day: 200 }],
  curve: [{ k: 10, recall: 0.2, baseline: 0.18 }, { k: 30, recall: 0.55, baseline: 0.5 }], note: "Review capacity K is per day.",
};

export const experiments: ExperimentRow[] = [{
  run_id: "test-run", name: "demo", started_at: "2026-02-01", git_commit: "abc123", folds: 3, seeds: 1, dataset: meta.dataset,
  headline: { pr_auc_xgb: 0.9, pr_auc_logreg: 0.85, recall5_p_only: 0.5, recall5_best: 0.55, best_method: "full|mult" },
}];

const VARIANTS = ["p_only", "p_a", "p_e", "all", "full"];
const MODES = ["mult", "linear", "constrained"];
const ab = (variant: string, mode: string, budget: number, recall: number) => ({
  variant, mode, budget, recall, recall_lo: recall - 0.05, recall_hi: recall + 0.05, precision: 0.3, redundancy: 0.2, beta: 0.5, gamma: 0.25, lam: 0.2,
});
export const curves = {
  run_id: "test-run", dataset: meta.dataset,
  ablation: VARIANTS.flatMap((v, i) => MODES.flatMap((m) => [0.01, 0.05, 0.1].map((b) => ab(v, m, b, 0.3 + i * 0.02 + b)))),
  baselines: { logreg: { pr_auc_mean: 0.85, pr_auc_std: 0.01, ci_lo: 0.8, ci_hi: 0.9 }, xgb_calibrated: { pr_auc_mean: 0.9, pr_auc_std: 0.01, ci_lo: null, ci_hi: null } },
  stability: { mean_fraud: 0.6, mean_legit: 0.55, mannwhitney_p: 0.2, kept_fraction: 0.9, hist_bins: [0, 0.5], hist_fraud: [3, 4], hist_legit: [5, 6] },
  faithfulness: { drop_topk: 0.2, drop_random: 0.05, faithful_frac: 0.7 }, efficiency: { ms_per_alert: 9, train_s: 3, rank_ms_all_methods: 100 },
  drift_pr_auc_thirds: [0.9, 0.88, 0.85], per_fold: [],
  recall_vs_budget: Object.fromEntries(VARIANTS.flatMap((v) => MODES.map((m) => [`${v}|${m}`, { "0.01": 0.2, "0.05": 0.5, "0.1": 0.7 }]))),
  curves: { fold: 3, pr: { xgb_calibrated: { precision: [1, 0.9, 0.8], recall: [0.1, 0.5, 0.9] } } },
} as unknown as Curves;
export const audit = {
  page: 1, size: 15, total: 2,
  items: [
    { id: 2, at: "2026-02-01T10:00:00", actor: "asha", action: "decision", detail: { alert_id: 101 } },
    { id: 1, at: "2026-02-01T09:59:00", actor: "asha", action: "view_alert", detail: { alert_id: 101 } },
  ],
};
