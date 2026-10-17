import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { METHOD_LABEL, METHOD_ORDER, type Method, type Mode, type SimResult, api } from "../api";
import { Card, Counter, ErrorBox, Segmented, Skeleton, Slider } from "../components/ui";
import { pct, useAsync, useDebounced } from "../lib";
import { useMeta } from "../meta";

const tip = { contentStyle: { background: "#0d1322", border: "1px solid #1f2a47", borderRadius: 10, fontSize: 12 } };

function Metric({ label, value, base, suffix = "%", digits = 1, good = "up" }: { label: string; value: number; base: number; suffix?: string; digits?: number; good?: "up" | "down" }) {
  const d = value - base, better = good === "up" ? d > 0.0005 : d < -0.0005, worse = good === "up" ? d < -0.0005 : d > 0.0005;
  return (
    <div className="glass hover px-4 py-3">
      <div className="text-[11px] uppercase tracking-[.08em] text-soft">{label}</div>
      <div className="mt-1 text-3xl font-semibold"><Counter value={value * 100} digits={digits} suffix={suffix} /></div>
      <div className={`mt-1 text-[12px] ${better ? "text-lo" : worse ? "text-hi" : "text-soft"}`}>
        {better ? "▲" : worse ? "▼" : "●"} {d >= 0 ? "+" : ""}{(d * 100).toFixed(1)} pts vs p only
      </div>
    </div>
  );
}

export default function SimulatorPage() {
  const meta = useMeta();
  const [method, setMethod] = useState<Method>("all");
  const [mode, setMode] = useState<Mode>("mult");
  const [k, setK] = useState(12);
  const [lam, setLam] = useState(0.3);
  const [beta, setBeta] = useState(0.5);
  const [gamma, setGamma] = useState(0.5);
  const [alpha, setAlpha] = useState(1);
  const [day, setDay] = useState("");
  const body = useDebounced({ method, mode, k, lam, beta, gamma, alpha, day: day || null }, 200);
  const sim = useAsync(() => api<SimResult>("/api/simulate", { method: "POST", json: body }), [JSON.stringify(body)]);
  const r = sim.data;
  const reset = (m: Method) => { setMethod(m); const w = meta.defaults[m]; if (w) { setBeta(w.beta); setGamma(w.gamma); setLam(w.lam); } };

  return (
    <div className="page-in grid gap-4 lg:grid-cols-[360px_1fr]">
      <Card title="Simulator" right={<button className="btn" onClick={() => reset(method)} title="Use the weights chosen on validation">Validated weights</button>}>
        <div className="grid gap-4">
          <div><div className="mb-1 text-[12px] text-soft">Method</div>
            <div className="flex flex-wrap gap-1.5">{METHOD_ORDER.map((m) => (
              <button key={m} className={`btn ${m === method ? "primary" : ""}`} aria-pressed={m === method} onClick={() => setMethod(m)}>{METHOD_LABEL[m]}</button>))}</div></div>
          <div><div className="mb-1 text-[12px] text-soft">Score form</div>
            <Segmented label="Score mode" value={mode} onChange={setMode} options={[{ value: "mult", label: "mult" }, { value: "linear", label: "linear" }, { value: "constrained", label: "constrained" }]} /></div>
          <Slider label="Review capacity K per day" value={k} min={1} max={200} onChange={setK} />
          <Slider label="α (linear only)" value={alpha} min={0} max={2} step={0.05} onChange={setAlpha} disabled={mode !== "linear"} format={(v) => v.toFixed(2)} />
          <Slider label="β anomaly" value={beta} min={0} max={2} step={0.05} onChange={setBeta} disabled={method === "p_only" || method === "p_e"} format={(v) => v.toFixed(2)} />
          <Slider label="γ stability" value={gamma} min={0} max={1} step={0.05} onChange={setGamma} disabled={method === "p_only" || method === "p_a"} format={(v) => v.toFixed(2)} />
          <Slider label="λ redundancy" value={lam} min={0} max={0.9} step={0.05} onChange={setLam} disabled={method !== "full"} format={(v) => v.toFixed(2)} hint="λ = 0 is plain top-K" />
          <label className="text-[12px] text-soft">Restrict to day
            <select className="input mt-1 w-full" value={day} onChange={(e) => setDay(e.target.value)} aria-label="Day"><option value="">all test days</option>{meta.days.map((d) => <option key={d}>{d}</option>)}</select></label>
        </div>
      </Card>

      <div className="grid content-start gap-4">
        {sim.error && <ErrorBox error={sim.error} retry={sim.reload} />}
        {!r && !sim.error && <Skeleton h={420} />}
        {r && (
          <>
            <div className={`grid grid-cols-2 gap-3 transition-opacity lg:grid-cols-4 ${sim.loading ? "opacity-60" : ""}`} aria-label="Live metrics">
              <Metric label={`Recall @ K=${r.k}/day`} value={r.recall} base={r.baseline.recall} />
              <Metric label="Precision" value={r.precision} base={r.baseline.precision} />
              <Metric label="Redundancy" value={r.redundancy} base={r.baseline.redundancy} digits={0} good="down" />
              <div className="glass hover px-4 py-3"><div className="text-[11px] uppercase tracking-[.08em] text-soft">Fraud caught</div>
                <div className="mt-1 text-3xl font-semibold"><Counter value={r.fraud_caught} /> <span className="text-base text-soft">/ {r.total_fraud}</span></div>
                <div className="mt-1 text-[12px] text-soft">p only: {r.baseline.fraud_caught} caught, {r.queued} queued</div></div>
            </div>
            <Card title="Recall as review capacity grows" right={<span className="chip">{r.days} day(s)</span>}>
              <div className="h-[260px]" role="img" aria-label="Recall versus capacity">
                <ResponsiveContainer><LineChart data={r.curve.map((c) => ({ k: c.k, current: c.recall * 100, baseline: c.baseline * 100 }))} margin={{ left: 0, right: 12, top: 6, bottom: 0 }}>
                  <CartesianGrid stroke="#1f2a47" /><XAxis dataKey="k" stroke="#8793b8" fontSize={11} label={{ value: "K per day", fill: "#8793b8", fontSize: 11, position: "insideBottom", offset: -2 }} />
                  <YAxis stroke="#8793b8" fontSize={11} width={38} tickFormatter={(v) => `${v}%`} /><Tooltip {...tip} formatter={(v) => `${Number(v).toFixed(1)}%`} /><Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="baseline" name="p only" stroke="#8793b8" strokeDasharray="5 4" dot={false} animationDuration={700} />
                  <Line dataKey="current" name={METHOD_LABEL[method]} stroke="#22d3ee" strokeWidth={2.4} animationDuration={700} />
                </LineChart></ResponsiveContainer>
              </div>
            </Card>
            <Card title="Per-day outcome">
              <div className="h-[200px]">
                <ResponsiveContainer><BarChart data={r.per_day.map((d) => ({ day: d.day.slice(5), caught: d.tp, missed: Math.max(d.fraud - d.tp, 0) }))} margin={{ left: 0, right: 8, top: 6, bottom: 0 }}>
                  <CartesianGrid stroke="#1f2a47" vertical={false} /><XAxis dataKey="day" stroke="#8793b8" fontSize={10} /><YAxis stroke="#8793b8" fontSize={10} width={28} allowDecimals={false} />
                  <Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="caught" stackId="a" fill="#34d399" animationDuration={700} /><Bar dataKey="missed" stackId="a" fill="#f43f5e55" animationDuration={700} />
                </BarChart></ResponsiveContainer>
              </div>
              <p className="mt-2 text-[11px] text-soft">{r.note} Current recall {pct(r.recall)}; baseline {pct(r.baseline.recall)}.</p>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
