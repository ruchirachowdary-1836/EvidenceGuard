import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, LabelList, Line, ReferenceDot, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { type Detail, type Verdict, api, setAnalyst, analyst } from "../api";
import { Card, ClusterTag, ErrorBox, Gauge, MiniBar, Skeleton, StabilityBar, toast } from "../components/ui";
import { clock, f3, go, money, pct, useAsync } from "../lib";

const FRIENDLY: Record<string, string> = {
  z_amt: "amount vs own history (robust z)", hour_dev: "hour vs usual hour", prod_rarity: "unusual product", cluster_cnt_86400: "device group activity 24h",
  cluster_cnt_3600: "device group activity 1h", new_device_: "new device", time_since_prev: "time since previous", amt_ratio_mean: "amount / own mean",
};
const label = (f: string) => FRIENDLY[f] ?? f.replace(/_/g, " ");

function ShapChart({ rows }: { rows: Detail["shap"] }) {
  const data = rows.slice(0, 10).map((r) => ({ ...r, name: label(r.feature), abs: Math.abs(r.shap) }));
  if (!data.length) return <div className="py-6 text-center text-xs text-soft">No attributions stored</div>;
  return (
    <div className="h-[310px]" role="img" aria-label="SHAP feature attributions">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 54, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke="#1f2a47" />
          <XAxis type="number" stroke="#8793b8" fontSize={11} />
          <YAxis type="category" dataKey="name" width={150} stroke="#8793b8" fontSize={11} tickLine={false} />
          <Tooltip contentStyle={{ background: "#0d1322", border: "1px solid #1f2a47", borderRadius: 10, fontSize: 12 }} formatter={(v) => [Number(v).toFixed(3), "SHAP (log-odds)"]} />
          <ReferenceLine x={0} stroke="#33457a" />
          <Bar dataKey="shap" radius={4} animationDuration={900}>
            {data.map((d) => <Cell key={d.feature} fill={d.shap >= 0 ? "#f43f5e" : "#38bdf8"} />)}
            <LabelList dataKey="shap" position="right" fontSize={10.5} fill="#8793b8" formatter={(v: unknown) => `${Number(v) >= 0 ? "▲ +" : "▼ "}${Number(v).toFixed(2)}`} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Which features were in the top-k under each perturbation; rows = features, columns = perturbations. */
function PerturbationHeatmap({ runs, base }: { runs: Detail["stability"]; base: string[] }) {
  const counts = new Map<string, number>();
  runs.forEach((r) => r.topk.forEach((f) => counts.set(f, (counts.get(f) ?? 0) + 1)));
  const feats = [...new Set([...base, ...counts.keys()])].sort((a, b) => (base.includes(b) ? 1 : 0) - (base.includes(a) ? 1 : 0) || (counts.get(b) ?? 0) - (counts.get(a) ?? 0)).slice(0, 11);
  const [hover, setHover] = useState<string>("");
  return (
    <div>
      <div className="overflow-x-auto">
        <div className="inline-grid gap-[3px]" style={{ gridTemplateColumns: `150px repeat(${runs.length}, 18px) 44px` }} role="table" aria-label="Perturbation heatmap">
          <div className="text-[10px] uppercase tracking-wider text-soft">feature \ perturbation</div>
          {runs.map((r) => <div key={r.m} className="text-center text-[9px] text-soft">{r.m + 1}</div>)}
          <div className="text-center text-[9px] text-soft">freq</div>
          {feats.map((f, fi) => (
            <div key={f} className="contents">
              <div className={`truncate text-[11.5px] ${base.includes(f) ? "font-semibold text-text" : "text-soft"}`} title={f}>{base.includes(f) ? "● " : "○ "}{label(f)}</div>
              {runs.map((r, ri) => {
                const on = r.topk.includes(f);
                return (
                  <div key={r.m} className={`pop h-[18px] rounded-[4px] ${on ? (base.includes(f) ? "bg-cyan" : "bg-hi") : "bg-line/60"} ${!r.kept ? "hatch" : ""}`}
                    style={{ animationDelay: `${(fi * runs.length + ri) * 6}ms`, outline: hover === `${f}-${r.m}` ? "2px solid #fff" : undefined }}
                    onMouseEnter={() => setHover(`${f}-${r.m}`)} onMouseLeave={() => setHover("")}
                    title={`${label(f)} | perturbation ${r.m + 1}: ${on ? "in top-k" : "not in top-k"} | Jaccard ${r.jaccard.toFixed(2)} | p shift ${r.prob_shift.toFixed(3)}${r.kept ? "" : " (discarded: not benign)"}`} />
                );
              })}
              <div className="num text-right text-[11px] text-soft">{counts.get(f) ?? 0}/{runs.length}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-[11px] text-soft">
        <span><i className="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-cyan" />original top-k stays</span>
        <span><i className="mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-hi" />intruder feature</span>
        <span><i className="hatch mr-1 inline-block h-2.5 w-2.5 rounded-sm bg-line" />hatched = discarded (moved p by more than delta)</span>
      </div>
      <div className="mt-3 flex items-end gap-[3px]" aria-label="Jaccard per perturbation">
        {runs.map((r) => (
          <div key={r.m} className="flex flex-col items-center gap-1" title={`perturbation ${r.m + 1}: Jaccard ${r.jaccard.toFixed(2)}`}>
            <div className="w-[18px] rounded-sm bg-violet/80" style={{ height: `${Math.max(3, r.jaccard * 36)}px`, opacity: r.kept ? 1 : 0.35 }} />
          </div>
        ))}
        <span className="ml-2 text-[10px] text-soft">Jaccard per perturbation</span>
      </div>
    </div>
  );
}

function BaselineChart({ d }: { d: Detail }) {
  const t0 = d.transaction.ts;
  const hist = d.baseline.history.map((h) => ({ x: +((h.ts - t0) / 86400).toFixed(2), amount: h.amount }));
  const hours = d.baseline.hours.map((c, h) => ({ h, c, cur: h === Math.floor(d.transaction.hour) }));
  const amt = d.transaction.amount;
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div>
        <div className="mb-1 text-[11px] text-soft">Past amounts of this customer ({d.baseline.n} txns) vs this transaction {d.baseline.median != null && <>| median {money(d.baseline.median)}</>}</div>
        <div className="h-[200px]" role="img" aria-label="Customer amount history">
          <ResponsiveContainer>
            <ComposedChart data={hist} margin={{ left: 0, right: 10, top: 10, bottom: 0 }}>
              <CartesianGrid stroke="#1f2a47" />
              <XAxis type="number" dataKey="x" domain={["dataMin", 0.5]} stroke="#8793b8" fontSize={10} tickFormatter={(v) => `${Math.round(v)}d`} />
              <YAxis stroke="#8793b8" fontSize={10} width={48} scale="log" domain={["auto", "auto"]} allowDataOverflow />
              <Tooltip contentStyle={{ background: "#0d1322", border: "1px solid #1f2a47", borderRadius: 10, fontSize: 12 }} formatter={(v) => money(Number(v))} labelFormatter={(v) => `${v} days before`} />
              {d.baseline.median != null && <ReferenceLine y={d.baseline.median} stroke="#22d3ee" strokeDasharray="4 4" label={{ value: "median", fill: "#22d3ee", fontSize: 10, position: "insideTopLeft" }} />}
              <Line dataKey="amount" stroke="#8b5cf655" dot={false} isAnimationActive />
              <Scatter dataKey="amount" fill="#8b5cf6" isAnimationActive />
              <ReferenceDot x={0} y={amt} r={7} fill="#f43f5e" stroke="#fff" strokeWidth={2} label={{ value: money(amt), fill: "#f43f5e", fontSize: 11, position: "top" }} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <div className="mb-1 text-[11px] text-soft">Usual hours of day vs this transaction ({clock(t0)})</div>
        <div className="h-[200px]" role="img" aria-label="Hour of day histogram">
          <ResponsiveContainer>
            <BarChart data={hours} margin={{ left: 0, right: 10, top: 10, bottom: 0 }}>
              <CartesianGrid stroke="#1f2a47" vertical={false} />
              <XAxis dataKey="h" stroke="#8793b8" fontSize={10} />
              <YAxis stroke="#8793b8" fontSize={10} width={26} allowDecimals={false} />
              <Tooltip contentStyle={{ background: "#0d1322", border: "1px solid #1f2a47", borderRadius: 10, fontSize: 12 }} formatter={(v) => [v, "past txns"]} labelFormatter={(h) => `${h}:00`} />
              <Bar dataKey="c" radius={3} animationDuration={800}>{hours.map((x) => <Cell key={x.h} fill={x.cur ? "#f43f5e" : "#8b5cf6"} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function DecisionForm({ id, onDone, history }: { id: number; onDone: () => void; history: Detail["decisions"] }) {
  const [who, setWho] = useState(analyst());
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<Verdict | null>(null);
  const submit = async (verdict: Verdict) => {
    setBusy(verdict);
    try {
      setAnalyst(who);
      await api(`/api/alerts/${id}/decision`, { method: "POST", json: { analyst: who || "analyst", verdict, note } });
      toast(`Decision recorded: ${verdict}`, "ok"); setNote(""); onDone();
    } catch (e) { toast((e as Error).message, "err"); } finally { setBusy(null); }
  };
  const BTN: { v: Verdict; text: string; cls: string }[] = [
    { v: "fraud", text: "▲ Confirm fraud", cls: "border-hi/60 text-hi" }, { v: "legit", text: "● Legitimate", cls: "border-lo/60 text-lo" },
    { v: "escalate", text: "◆ Escalate", cls: "border-violet/60 text-violet" }, { v: "needs_info", text: "? Need info", cls: "border-mid/60 text-mid" },
  ];
  return (
    <div className="grid gap-3">
      <div className="grid gap-2 sm:grid-cols-[160px_1fr]">
        <input className="input" aria-label="Analyst name" value={who} onChange={(e) => setWho(e.target.value)} maxLength={40} placeholder="Analyst" />
        <input className="input" aria-label="Decision note" value={note} onChange={(e) => setNote(e.target.value)} maxLength={500} placeholder="Note (optional): why this verdict?" />
      </div>
      <div className="flex flex-wrap gap-2">
        {BTN.map((b) => <button key={b.v} className={`btn ${b.cls}`} disabled={busy !== null} onClick={() => submit(b.v)}>{busy === b.v ? "Saving..." : b.text}</button>)}
      </div>
      {history.length > 0 && (
        <ul className="grid gap-1.5 text-[12px]" aria-label="Decision history">
          {history.map((h, i) => <li key={i} className="row-in rounded-lg border border-line bg-panel2/60 px-3 py-1.5"><b className="text-violet">{h.verdict}</b> by {h.analyst} <span className="text-soft">{h.note && `- ${h.note}`}</span></li>)}
        </ul>
      )}
    </div>
  );
}

export default function DetailPage({ id }: { id: number }) {
  const [reveal, setReveal] = useState(false);
  const st = useAsync(() => api<Detail>(`/api/alerts/${id}${reveal ? "?reveal=true" : ""}`), [id, reveal]);
  const d = st.data;
  const base = useMemo(() => d?.shap.slice(0, 5).map((s) => s.feature) ?? [], [d]);
  if (st.error) return <ErrorBox error={st.error} retry={st.reload} />;
  if (!d) return <div className="grid gap-4"><Skeleton h={160} /><Skeleton h={320} /></div>;
  const t = d.transaction;
  const facts = Object.entries(t.facts).filter(([k]) => !["mean_amt"].includes(k));
  return (
    <div className="page-in grid gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <button className="btn" onClick={() => go("#/queue")}>{"←"} Queue</button>
        <h1 className="text-lg font-semibold">Alert #{d.alert_id} <span className="text-soft">- transaction #{t.txn_id}</span></h1>
        <ClusterTag id={t.cluster_key} size={d.peers.length + 1} />
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-[12px] text-soft">
          <input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} /> Reveal ground-truth label (evaluation only)
        </label>
        {reveal && t.label != null && <span className={`chip ${t.label ? "border-hi/60 text-hi" : "border-lo/60 text-lo"}`}>label: {t.label ? "fraud" : "legitimate"}</span>}
      </div>

      <div className="grid gap-4 lg:grid-cols-[auto_1fr_1.2fr]">
        <Card className="grid place-items-center"><Gauge p={d.scores.p} /></Card>
        <Card title="Scores">
          <div className="grid gap-2 text-[12.5px]">
            <div className="flex items-center justify-between"><span className="text-soft">Fraud probability p</span><span className="num">{pct(d.scores.p)}</span></div>
            <div className="flex items-center justify-between"><span className="text-soft">Anomaly a (vs this customer)</span><MiniBar v={d.scores.a} /></div>
            <div className="flex items-center justify-between"><span className="text-soft">Explanation stability e</span><StabilityBar e={d.scores.e} /></div>
            <div className="flex items-center justify-between"><span className="text-soft">R multiplicative</span><span className="num">{f3(d.scores.r_mult)}</span></div>
            <div className="flex items-center justify-between"><span className="text-soft">R linear (baseline)</span><span className="num">{f3(d.scores.r_linear)}</span></div>
            <p className="mt-1 text-[11px] text-soft">Stable is not the same as correct: faithfulness is tested separately with a deletion test (see Models).</p>
          </div>
        </Card>
        <Card title="Transaction facts">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[12.5px] sm:grid-cols-3">
            <div><dt className="text-[10.5px] uppercase tracking-wider text-soft">amount</dt><dd className="num">{money(t.amount)}</dd></div>
            <div><dt className="text-[10.5px] uppercase tracking-wider text-soft">day / time</dt><dd className="num">{d.day} {clock(t.ts)}</dd></div>
            <div><dt className="text-[10.5px] uppercase tracking-wider text-soft">entity (proxy)</dt><dd className="num truncate" title={t.entity_key}>{t.entity_key}</dd></div>
            {facts.map(([k, v]) => <div key={k}><dt className="text-[10.5px] uppercase tracking-wider text-soft">{k.replace(/_/g, " ")}</dt><dd className="num">{typeof v === "number" ? Number(v.toFixed(2)) : v}</dd></div>)}
          </dl>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Why was this flagged? SHAP attributions" right={<span className="chip">top-5 form the explanation</span>}><ShapChart rows={d.shap} /></Card>
        <Card title="Is the explanation stable?" right={<StabilityBar e={d.scores.e} wide />}>
          {d.stability.length ? <PerturbationHeatmap runs={d.stability} base={base} /> : <div className="py-6 text-center text-xs text-soft">No perturbation runs stored</div>}
        </Card>
      </div>

      <Card title="Customer baseline vs this transaction"><BaselineChart d={d} /></Card>

      <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <Card title={`Peers in cluster (±48 h) - ${d.peers.length}`}>
          {d.peers.length === 0 ? <div className="py-5 text-center text-xs text-soft">No other transactions from this device group nearby.</div> : (
            <div className="max-h-[280px] overflow-auto">
              <table className="dense"><thead><tr><th>Txn</th><th>Time</th><th>Entity</th><th>Amount</th><th>p</th><th>e</th>{reveal && <th>label</th>}</tr></thead>
                <tbody>{d.peers.map((p) => (
                  <tr key={p.txn_id} onClick={() => p.alert_id && go(`#/alert/${p.alert_id}`)} style={{ cursor: p.alert_id ? "pointer" : "default" }}>
                    <td className="num">#{p.txn_id}</td><td className="num">{clock(p.ts)}</td><td className="num">{p.entity_key}</td><td className="num">{money(p.amount)}</td>
                    <td className="num">{p.p == null ? "-" : pct(p.p, 0)}</td><td className="num">{p.e == null ? "-" : p.e.toFixed(2)}</td>{reveal && <td>{p.label ? "fraud" : "legit"}</td>}
                  </tr>))}</tbody></table>
            </div>
          )}
        </Card>
        <Card title="Analyst decision"><DecisionForm id={id} history={d.decisions} onDone={st.reload} /></Card>
      </div>
    </div>
  );
}
