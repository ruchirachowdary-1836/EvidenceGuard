// Typed API client for the EvidenceGuard FastAPI backend.
export type Method = "p_only" | "p_a" | "p_e" | "all" | "full";
export type Mode = "mult" | "linear" | "constrained";
export type Verdict = "fraud" | "legit" | "escalate" | "needs_info";

export const METHOD_LABEL: Record<Method, string> = {
  p_only: "p only", p_a: "p + anomaly", p_e: "p + stability", all: "all components", full: "full + queue",
};
export const METHOD_ORDER: Method[] = ["p_only", "p_a", "p_e", "all", "full"];

export interface Meta {
  run_id: string; model_version: string; caveat: string; days: string[]; default_day: string;
  methods: Method[]; modes: Mode[]; defaults: Partial<Record<Method, { beta: number; gamma: number; lam: number }>>;
  pool_size: number; n_day: Record<string, number>; default_budget: number;
  dataset: { source: string; rows: number; fraud_rate: number; entities: number; caveat: string };
}

export interface AlertRow {
  alert_id: number; txn_id: number; queue_rank: number; p: number; a: number; e: number; score: number;
  top_features: [string, number][]; cluster_key: string; cluster_size: number; amount: number; ts: number; verdict: string | null;
}
export interface Queue {
  run_id: string; day: string; method: Method; k: number; lam: number; weights: Record<string, number>;
  n_pool: number; n_day: number; redundancy: number; alerts: AlertRow[];
}
export interface ShapRow { feature: string; shap: number; rank: number; value: number }
export interface StabRun { m: number; jaccard: number; prob_shift: number; kept: boolean; topk: string[] }
export interface Detail {
  alert_id: number; run_id: string; day: string;
  transaction: { txn_id: number; entity_key: string; cluster_key: string; ts: number; amount: number; hour: number; label: number | null; facts: Record<string, number | string> };
  scores: { p: number; a: number; e: number; r_linear: number; r_mult: number };
  shap: ShapRow[]; stability: StabRun[];
  baseline: { history: { ts: number; amount: number }[]; median: number | null; mad: number | null; hours: number[]; n: number };
  peers: { alert_id: number | null; txn_id: number; ts: number; amount: number; entity_key: string; p: number | null; e: number | null; label: number | null }[];
  decisions: { analyst: string; verdict: string; note: string; decided_at: string }[];
}
export interface SimResult {
  run_id: string; k: number; days: number; total_fraud: number; recall: number; precision: number; redundancy: number; fraud_caught: number;
  queued: number; delta_recall: number; note: string;
  baseline: { recall: number; precision: number; redundancy: number; fraud_caught: number };
  per_day: { day: string; k: number; tp: number; fraud: number; n_day: number }[];
  curve: { k: number; recall: number; baseline: number }[];
}
export interface AblationRow {
  variant: Method; mode: Mode; weights?: "validated" | "fixed"; budget: number; recall: number; recall_lo: number; recall_hi: number; precision: number;
  redundancy: number; beta: number; gamma: number; lam: number;
}
export interface Curves {
  run_id: string; ablation: AblationRow[];
  baselines: Record<string, { pr_auc_mean: number; pr_auc_std: number; ci_lo: number | null; ci_hi: number | null }>;
  stability: Record<string, number | number[]>; faithfulness: Record<string, number>; efficiency: Record<string, number>;
  drift_pr_auc_thirds: number[]; per_fold: Record<string, number | string>[];
  recall_vs_budget: Record<string, Record<string, number>>;
  curves: { pr: Record<string, { precision: number[]; recall: number[] }>; fold: number };
  dataset: Meta["dataset"];
}
export interface ExperimentRow {
  run_id: string; name: string; started_at: string; git_commit: string; folds: number; seeds: number; dataset: Meta["dataset"];
  headline: { pr_auc_xgb: number | null; pr_auc_logreg: number | null; recall5_p_only: number | null; recall5_best: number | null; best_method: string | null };
}
export interface AuditPage {
  page: number; size: number; total: number; items: { id: number; at: string; actor: string; action: string; detail: Record<string, unknown> }[];
}

export function analyst(): string {
  try { return localStorage.getItem("eg.analyst") || "analyst"; } catch { return "analyst"; }
}
export function setAnalyst(v: string): void {
  try { localStorage.setItem("eg.analyst", v.slice(0, 40) || "analyst"); } catch { /* storage blocked */ }
}

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { "X-Analyst": analyst(), ...(init.headers as Record<string, string> | undefined) };
  let body = init.body;
  if (init.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(init.json); }
  const r = await fetch(path, { ...init, headers, body });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const qs = (o: Record<string, string | number | boolean | undefined | null>): string => {
  const p = new URLSearchParams();
  Object.entries(o).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") p.set(k, String(v)); });
  const s = p.toString();
  return s ? `?${s}` : "";
};
