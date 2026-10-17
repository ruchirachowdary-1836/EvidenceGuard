import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { type Curves, type ExperimentRow, METHOD_LABEL, METHOD_ORDER, type Mode, api } from "../api";
import { Card, Counter, Empty, ErrorBox, Segmented, Skeleton } from "../components/ui";
import { f3, pct, useAsync } from "../lib";

const COL: Record<string, string> = { p_only: "#8793b8", p_a: "#38bdf8", p_e: "#fbbf24", all: "#8b5cf6", full: "#22d3ee", logreg: "#fbbf24", xgb: "#8b5cf6", xgb_weighted: "#38bdf8", xgb_calibrated: "#22d3ee" };
const tip = { contentStyle: { background: "#0d1322", border: "1px solid #1f2a47", borderRadius: 10, fontSize: 12 } };

export default function ModelsPage() {
  const runs = useAsync(() => api<ExperimentRow[]>("/api/experiments"), []);
  const [runId, setRunId] = useState<string>("");
  const [budget, setBudget] = useState(0.05);
  const [mode, setMode] = useState<Mode>("mult");
  const [wt, setWt] = useState<"validated" | "fixed">("validated");
  const id = runId || runs.data?.[0]?.run_id;
  const cur = useAsync(() => (id ? api<Curves>(`/api/experiments/${id}/curves`) : Promise.resolve(undefined)), [id]);
  const c = cur.data;

  const prData = useMemo(() => {
    if (!c) return [];
    const out: Record<string, number>[] = [];
    Object.entries(c.curves.pr).forEach(([k, v]) => v.recall.forEach((r, i) => out.push({ recall: r, [k]: v.precision[i] })));
    return out.sort((a, b) => a.recall - b.recall);
  }, [c]);
  const budgets = useMemo(() => (c ? [...new Set(c.ablation.map((r) => r.budget))].sort((a, b) => a - b) : []), [c]);
  const rows = useMemo(() => (c ? c.ablation.filter((r) => r.budget === budget && (r.variant === "p_only" || (r.mode === mode && (r.weights ?? "validated") === wt))) : []), [c, budget, mode, wt]);
  const budgetCurve = useMemo(() => {
    if (!c) return [];
    const keys = METHOD_ORDER.map((m) => [m, m === "p_only" ? "p_only|mult" : `${m}|${mode}${wt === "fixed" ? "|fixed" : ""}`] as const);
    const xs = Object.keys(c.recall_vs_budget["p_only|mult"] ?? {}).sort((a, b) => +a - +b);
    return xs.map((x) => ({ budget: +x * 100, ...Object.fromEntries(keys.map(([m, k]) => [m, (c.recall_vs_budget[k]?.[x] ?? 0) * 100])) }));
  }, [c, mode, wt]);
  const st = c?.stability as Record<string, number & number[]> | undefined;
  const hist = useMemo(() => {
    if (!st) return [];
    const bins = st.hist_bins as unknown as number[], hf = st.hist_fraud as unknown as number[], hl = st.hist_legit as unknown as number[];
    const nf = hf.reduce((a, b) => a + b, 0) || 1, nl = hl.reduce((a, b) => a + b, 0) || 1;
    return hf.map((v, i) => ({ bin: bins[i].toFixed(2), fraud: (100 * v) / nf, legit: (100 * hl[i]) / nl }));
  }, [st]);
  const best = rows.filter((r) => r.variant !== "p_only").sort((a, b) => b.recall - a.recall)[0];
  const base = rows.find((r) => r.variant === "p_only");
  const gain = best && base ? best.recall - base.recall : 0;
  const overlap = best && base ? best.recall_lo <= base.recall_hi && base.recall_lo <= best.recall_hi : true;

  if (runs.error) return <ErrorBox error={runs.error} retry={runs.reload} />;
  if (!runs.data || (id && !c && !cur.error)) return <div className="grid gap-4"><Skeleton h={90} /><Skeleton h={320} /><Skeleton h={320} /></div>;
  if (!runs.data.length) return <Empty title="No experiment runs loaded" hint="Run `make demo` (small) or `make ablate` to populate the registry." />;
  if (cur.error) return <ErrorBox error={cur.error} retry={cur.reload} />;
  if (!c || !st) return <Empty title="Select a run" />;

  return (
    <div className="page-in grid gap-4">
      <Card title="Experiment registry" right={<select className="input" aria-label="Run" value={id} onChange={(e) => setRunId(e.target.value)}>{runs.data.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_id}</option>)}</select>}>
        <div className="grid grid-cols-2 gap-3 text-[12.5px] md:grid-cols-4">
          <div><div className="text-soft">Dataset</div><div className="font-medium">{c.dataset?.source} <span className="text-soft">({c.dataset?.rows?.toLocaleString()} rows)</span></div></div>
          <div><div className="text-soft">Protocol</div><div className="font-medium">{runs.data.find((r) => r.run_id === id)?.folds} folds x {runs.data.find((r) => r.run_id === id)?.seeds} seed(s)</div></div>
          <div><div className="text-soft">Commit</div><div className="num">{runs.data.find((r) => r.run_id === id)?.git_commit}</div></div>
          <div><div className="text-soft">Ranker latency</div><div className="num"><Counter value={c.efficiency.ms_per_alert ?? 0} digits={1} /> ms / alert</div></div>
        </div>
        <p className="mt-3 rounded-lg border border-mid/40 bg-mid/10 px-3 py-2 text-[12px] text-mid" role="note">{c.dataset?.caveat} Numbers describe this dataset only and are not evidence about real UPI traffic.</p>
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Baselines: PR-AUC on test (primary metric)">
          <table className="dense"><thead><tr><th>Model</th><th>PR-AUC</th><th>95% CI</th><th>sd over runs</th></tr></thead>
            <tbody>{Object.entries(c.baselines).map(([k, v], i) => (
              <tr key={k} className="row-in" style={{ animationDelay: `${i * 50}ms`, cursor: "default" }}><td><i className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: COL[k] }} />{k}</td>
                <td className="num font-semibold">{f3(v.pr_auc_mean)}</td><td className="num text-soft">{v.ci_lo != null ? `${f3(v.ci_lo)} - ${f3(v.ci_hi)}` : "-"}</td><td className="num text-soft">{f3(v.pr_auc_std)}</td></tr>))}</tbody></table>
          <p className="mt-2 text-[11px] text-soft">Accuracy is never a headline number. The main XGBoost variant is chosen on validation, then isotonic-calibrated on validation.</p>
        </Card>
        <Card title="Precision-recall curves (last fold, test)">
          <div className="h-[250px]" role="img" aria-label="Precision recall curves">
            <ResponsiveContainer><LineChart data={prData} margin={{ left: 0, right: 12, top: 6, bottom: 0 }}>
              <CartesianGrid stroke="#1f2a47" /><XAxis type="number" dataKey="recall" domain={[0, 1]} stroke="#8793b8" fontSize={11} tickFormatter={(v) => v.toFixed(1)} label={{ value: "recall", fill: "#8793b8", fontSize: 11, position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} stroke="#8793b8" fontSize={11} width={34} /><Tooltip {...tip} formatter={(v) => Number(v).toFixed(3)} /><Legend wrapperStyle={{ fontSize: 11 }} />
              {Object.keys(c.curves.pr).map((k) => <Line key={k} dataKey={k} stroke={COL[k]} dot={false} strokeWidth={2} connectNulls animationDuration={1000} />)}
            </LineChart></ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card title="Ablation: variants 1-5 on held-out test windows" right={<div className="flex flex-wrap items-center gap-2">
        <Segmented label="Budget" value={String(budget)} onChange={(v) => setBudget(Number(v))} options={budgets.map((b) => ({ value: String(b), label: `${b * 100}%` }))} />
        <Segmented label="Weights" value={wt} onChange={setWt} options={[{ value: "validated", label: "validated" }, { value: "fixed", label: "fixed 0.5/0.5/0.3" }]} />
        <Segmented label="Score mode" value={mode} onChange={setMode} options={[{ value: "mult", label: "multiplicative" }, { value: "linear", label: "linear" }, { value: "constrained", label: "constrained" }]} />
      </div>}>
        <div className="overflow-x-auto"><table className="dense"><thead><tr><th>Variant</th><th>Recall@K</th><th>95% CI</th><th>Precision</th><th>Redundancy</th><th>weights (beta, gamma, lambda)</th></tr></thead>
          <tbody>{rows.map((r, i) => (
            <tr key={`${r.variant}${r.mode}${r.budget}`} className="row-in" style={{ animationDelay: `${i * 60}ms`, cursor: "default" }}>
              <td><i className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: COL[r.variant] }} />{i + 1}. {METHOD_LABEL[r.variant]}</td>
              <td><div className="flex items-center gap-2"><div className="h-1.5 w-24 overflow-hidden rounded-full bg-line"><div className="fill h-full" style={{ width: `${r.recall * 100}%`, background: COL[r.variant] }} /></div><b className="num">{pct(r.recall)}</b></div></td>
              <td className="num text-soft">{pct(r.recall_lo)} - {pct(r.recall_hi)}</td><td className="num">{pct(r.precision)}</td><td className="num">{pct(r.redundancy, 0)}</td>
              <td className="num text-soft">{r.beta.toFixed(2)}, {r.gamma.toFixed(2)}, {r.lam.toFixed(2)}</td></tr>))}</tbody></table></div>
        {best && base && (
          <p className={`mt-3 rounded-lg border px-3 py-2 text-[12.5px] ${gain > 0 && !overlap ? "border-lo/40 bg-lo/10 text-lo" : "border-mid/40 bg-mid/10 text-mid"}`} role="status">
            {gain > 0 && !overlap
              ? `Best non-baseline variant (${METHOD_LABEL[best.variant]}) beats p only by ${(gain * 100).toFixed(1)} points and the confidence intervals do not overlap.`
              : gain > 0
                ? `Best non-baseline variant (${METHOD_LABEL[best.variant]}) is ${(gain * 100).toFixed(1)} points above p only, but the confidence intervals overlap: not a reliable improvement.`
                : "No variant beat the p-only baseline at this budget. Weights are chosen on validation only; this negative result is reported as is."}
          </p>
        )}
      </Card>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Recall vs review budget (all folds)">
          <div className="h-[260px]" role="img" aria-label="Recall versus budget">
            <ResponsiveContainer><LineChart data={budgetCurve} margin={{ left: 0, right: 12, top: 6, bottom: 0 }}>
              <CartesianGrid stroke="#1f2a47" /><XAxis dataKey="budget" stroke="#8793b8" fontSize={11} tickFormatter={(v) => `${v}%`} /><YAxis stroke="#8793b8" fontSize={11} width={34} tickFormatter={(v) => `${v}%`} />
              <Tooltip {...tip} formatter={(v) => `${Number(v).toFixed(1)}%`} labelFormatter={(v) => `budget ${v}%`} /><Legend wrapperStyle={{ fontSize: 11 }} />
              {METHOD_ORDER.map((m) => <Line key={m} dataKey={m} name={METHOD_LABEL[m]} stroke={COL[m]} strokeWidth={m === "p_only" ? 1.5 : 2.2} strokeDasharray={m === "p_only" ? "5 4" : undefined} dot={false} animationDuration={1100} />)}
            </LineChart></ResponsiveContainer>
          </div>
        </Card>
        <Card title="RQ1: explanation stability, fraud vs legitimate" right={<span className="chip">Mann-Whitney p = {st.mannwhitney_p != null ? Number(st.mannwhitney_p).toFixed(3) : "-"}</span>}>
          <div className="h-[200px]" role="img" aria-label="Stability distribution">
            <ResponsiveContainer><BarChart data={hist} margin={{ left: 0, right: 8, top: 6, bottom: 0 }}>
              <CartesianGrid stroke="#1f2a47" vertical={false} /><XAxis dataKey="bin" stroke="#8793b8" fontSize={10} interval={3} /><YAxis stroke="#8793b8" fontSize={10} width={30} tickFormatter={(v) => `${v}%`} />
              <Tooltip {...tip} formatter={(v) => `${Number(v).toFixed(1)}%`} /><Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="legit" name="legitimate" fill="#38bdf8" radius={2} animationDuration={900} /><Bar dataKey="fraud" fill="#f43f5e" radius={2} animationDuration={900} />
            </BarChart></ResponsiveContainer>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 text-center text-[12px]">
            <div className="glass px-2 py-2"><div className="text-soft">mean e fraud</div><div className="num text-base">{f3(st.mean_fraud)}</div></div>
            <div className="glass px-2 py-2"><div className="text-soft">mean e legit</div><div className="num text-base">{f3(st.mean_legit)}</div></div>
            <div className="glass px-2 py-2"><div className="text-soft">benign kept</div><div className="num text-base">{pct(st.kept_fraction, 0)}</div></div>
          </div>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card title="Faithfulness (deletion test)"><div className="grid gap-1 text-[12.5px]">
          <div className="flex justify-between"><span className="text-soft">p drop, delete top-k SHAP</span><b className="num">{f3(c.faithfulness.drop_topk)}</b></div>
          <div className="flex justify-between"><span className="text-soft">p drop, delete random k</span><b className="num">{f3(c.faithfulness.drop_random)}</b></div>
          <div className="flex justify-between"><span className="text-soft">top-k hurts more in</span><b className="num">{pct(c.faithfulness.faithful_frac, 0)}</b></div>
          <p className="mt-1 text-[11px] text-soft">Stable is not the same as correct.</p></div></Card>
        <Card title="Drift (PR-AUC by test third)"><div className="flex items-end gap-3 pt-2">{c.drift_pr_auc_thirds.map((v, i) => (
          <div key={i} className="flex-1 text-center"><div className="mx-auto w-10 rounded-t bg-gradient-to-t from-violet to-cyan fill" style={{ height: `${Math.max(6, v * 80)}px`, transformOrigin: "bottom" }} /><div className="num mt-1 text-[11px]">{f3(v)}</div><div className="text-[10px] text-soft">third {i + 1}</div></div>))}</div></Card>
        <Card title="Efficiency"><div className="grid gap-1 text-[12.5px]">
          <div className="flex justify-between"><span className="text-soft">SHAP + perturbations</span><b className="num">{(c.efficiency.ms_per_alert ?? 0).toFixed(1)} ms/alert</b></div>
          <div className="flex justify-between"><span className="text-soft">Training (3 models)</span><b className="num">{(c.efficiency.train_s ?? 0).toFixed(1)} s</b></div>
          <div className="flex justify-between"><span className="text-soft">Ranking, all variants</span><b className="num">{(c.efficiency.rank_ms_all_methods ?? 0).toFixed(0)} ms</b></div></div></Card>
      </div>
    </div>
  );
}
