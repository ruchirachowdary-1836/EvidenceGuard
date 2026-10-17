import { useEffect, useMemo, useRef, useState } from "react";
import { type AlertRow, METHOD_LABEL, METHOD_ORDER, type Method, type Mode, type Queue, api, qs } from "../api";
import { Card, ClusterTag, Counter, Empty, ErrorBox, MiniBar, RiskBar, Segmented, Skeleton, Slider, Stat, StabilityBar, useOnKey } from "../components/ui";
import { clock, go, money, riskLevel, useAsync, useDebounced } from "../lib";
import { useMeta } from "../meta";

interface LiveTxn { alert_id: number; txn_id: number; p: number; e: number; amount: number; ts: number; day: string; cluster_key: string; n: number }

function LiveTicker() {
  const [items, setItems] = useState<LiveTxn[]>([]);
  const [paused, setPaused] = useState(false);
  const [status, setStatus] = useState<"connecting" | "live" | "offline">("connecting");
  const pausedRef = useRef(paused);
  pausedRef.current = paused;
  useEffect(() => {
    if (typeof EventSource === "undefined") { setStatus("offline"); return; }
    const es = new EventSource("/api/stream?interval=1.6");
    let n = 0;
    es.addEventListener("hello", () => setStatus("live"));
    es.addEventListener("txn", (ev) => {
      if (pausedRef.current) return;
      const d = JSON.parse((ev as MessageEvent).data);
      setItems((x) => [{ ...d, n: ++n }, ...x].slice(0, 8));
    });
    es.onerror = () => setStatus("offline");
    return () => es.close();
  }, []);
  return (
    <Card title={<span className="flex items-center gap-2"><i className={`inline-block h-2 w-2 rounded-full ${status === "live" && !paused ? "pulse-dot bg-cyan" : "bg-soft"}`} />Live candidate feed <span className="chip">held-out replay, not live traffic</span></span>}
      right={<button className="btn" onClick={() => setPaused((p) => !p)} aria-pressed={paused}>{paused ? "Resume" : "Pause"}</button>}>
      {items.length === 0 ? (
        <div className="py-3 text-center text-xs text-soft">{status === "offline" ? "Stream unavailable" : "Waiting for the next scored candidate..."}</div>
      ) : (
        <div className="flex gap-3 overflow-hidden" aria-label="Incoming candidate alerts">
          {items.map((t) => {
            const lvl = riskLevel(t.p);
            return (
              <button key={t.n} onClick={() => go(`#/alert/${t.alert_id}`)} className="slide-in glass hover min-w-[158px] px-3 py-2 text-left"
                style={{ borderColor: lvl === "hi" ? "#f43f5e88" : undefined }}>
                <div className="flex items-center justify-between text-[11px] text-soft"><span className="num">#{t.txn_id}</span><span>{clock(t.ts)}</span></div>
                <div className="mt-1 flex items-baseline justify-between"><span className="num text-lg">{(t.p * 100).toFixed(0)}%</span><span className="num text-xs text-soft">{money(t.amount)}</span></div>
                <div className="mt-1 truncate text-[10.5px] text-soft">{t.cluster_key}</div>
              </button>
            );
          })}
        </div>
      )}
    </Card>
  );
}

export default function QueuePage() {
  const meta = useMeta();
  const [day, setDay] = useState(meta.default_day);
  const [method, setMethod] = useState<Method>("full");
  const [mode, setMode] = useState<Mode>("mult");
  const nDay = meta.n_day[day] ?? 100;
  const defK = Math.max(1, Math.ceil(meta.default_budget * nDay));
  const [k, setK] = useState(defK);
  const [lam, setLam] = useState<number | null>(null);
  const [beta, setBeta] = useState<number | null>(null);
  const [gamma, setGamma] = useState<number | null>(null);
  const [sel, setSel] = useState(0);
  useEffect(() => { setK(Math.max(1, Math.ceil(meta.default_budget * (meta.n_day[day] ?? 100)))); }, [day, meta]);

  const dk = useDebounced(k, 220), dl = useDebounced(lam, 220), db = useDebounced(beta, 220), dg = useDebounced(gamma, 220);
  const queue = useAsync(
    () => Promise.all([
      api<Queue>(`/api/alerts${qs({ day, method, mode, k: dk, lam: dl, beta: db, gamma: dg })}`),
      api<Queue>(`/api/alerts${qs({ day, method: "p_only", k: dk })}`),
    ]),
    [day, method, mode, dk, dl, db, dg],
  );
  const q = queue.data?.[0];
  const base = queue.data?.[1];
  const baseRank = useMemo(() => new Map(base?.alerts.map((a) => [a.txn_id, a.queue_rank]) ?? []), [base]);
  const w = q?.weights;
  const validatedZero = (meta.defaults[method]?.beta ?? 0) === 0 && (meta.defaults[method]?.gamma ?? 0) === 0 && method !== "p_only";
  const maxK = Math.min(300, Math.max(5, q?.n_pool ?? 100));

  useOnKey((e) => {
    if (!q?.alerts.length) return;
    if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(q.alerts.length - 1, s + 1)); }
    if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
    if (e.key === "Enter") go(`#/alert/${q.alerts[sel]?.alert_id}`);
  });
  useEffect(() => { setSel(0); }, [day, method, mode, dk]);
  useEffect(() => { document.getElementById(`row-${sel}`)?.scrollIntoView({ block: "nearest" }); }, [sel]);

  const avgE = q?.alerts.length ? q.alerts.reduce((s, a) => s + a.e, 0) / q.alerts.length : 0;
  const decided = q?.alerts.filter((a) => a.verdict).length ?? 0;
  const version = `${day}|${method}|${mode}|${dk}|${dl}|${db}|${dg}`;

  return (
    <div className="page-in grid gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Review budget K" value={q?.k ?? 0} hint={`${((q?.k ?? 0) / Math.max(nDay, 1) * 100).toFixed(1)}% of ${nDay} txns that day`} />
        <Stat label="Candidate pool" value={q?.n_pool ?? 0} hint="scored for explanation stability" />
        <Stat label="Redundancy" value={(q?.redundancy ?? 0) * 100} digits={0} suffix="%" hint="alerts sharing a cluster in the queue" tone={(q?.redundancy ?? 0) > 0.5 ? "text-mid" : "text-text"} />
        <Stat label="Mean stability" value={avgE} digits={2} hint="top-5 SHAP Jaccard" />
        <Stat label="Decided" value={decided} hint={`of ${q?.alerts.length ?? 0} in queue`} tone="text-lo" />
      </div>

      <Card title="Budgeted queue" right={<span className="chip">run {meta.run_id}</span>}>
        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr_1fr]">
          <div className="grid gap-3">
            <div className="flex flex-wrap items-center gap-3">
              <Segmented label="Ranking method" value={method} onChange={setMethod} options={METHOD_ORDER.map((m) => ({ value: m, label: METHOD_LABEL[m] }))} />
            </div>
            <div className="flex flex-wrap items-center gap-3 text-[12px] text-soft">
              <label>Day <select className="input ml-1" value={day} onChange={(e) => setDay(e.target.value)} aria-label="Day">{meta.days.map((d) => <option key={d}>{d}</option>)}</select></label>
              <label>Score <select className="input ml-1" value={mode} onChange={(e) => setMode(e.target.value as Mode)} aria-label="Score mode">{meta.modes.map((m) => <option key={m} value={m}>{m === "mult" ? "multiplicative" : m}</option>)}</select></label>
            </div>
          </div>
          <div className="grid gap-3">
            <Slider label="Capacity K / day" value={k} min={1} max={maxK} onChange={setK} hint="alerts an analyst team can review" />
            <Slider label="Redundancy penalty λ" value={lam ?? q?.lam ?? 0.3} min={0} max={0.9} step={0.05} onChange={setLam} disabled={method !== "full"} format={(v) => v.toFixed(2)} hint={method === "full" ? "extra alert from a cluster is discounted by (1 - λ)" : "only for the full method"} />
          </div>
          <div className="grid gap-3">
            <Slider label="Anomaly weight β" value={beta ?? w?.beta ?? 0.5} min={0} max={2} step={0.05} onChange={setBeta} disabled={method === "p_only" || method === "p_e"} format={(v) => v.toFixed(2)} />
            <Slider label="Stability weight γ" value={gamma ?? w?.gamma ?? 0.5} min={0} max={1} step={0.05} onChange={setGamma} disabled={method === "p_only" || method === "p_a"} format={(v) => v.toFixed(2)} />
          </div>
        </div>
        {validatedZero && beta == null && gamma == null && (
          <p className="mt-3 rounded-lg border border-mid/40 bg-mid/10 px-3 py-2 text-[12px] text-mid" role="note">
            Validation selected β = γ = 0 for this method: anomaly and stability did not improve recall on the validation window, so this queue equals "p only". Move the weight sliders to explore anyway; the Models page reports the honest result.
          </p>
        )}
      </Card>

      <LiveTicker />

      <Card title={`Ranked alerts for ${day}`} right={<span className="chip">{METHOD_LABEL[method]}</span>}>
        {queue.error && <ErrorBox error={queue.error} retry={queue.reload} />}
        {!q && !queue.error && <Skeleton h={320} />}
        {q && q.alerts.length === 0 && <Empty title="No alerts for this day" hint="Pick another day or raise the budget." />}
        {q && q.alerts.length > 0 && (
          <div className={`max-h-[560px] overflow-auto transition-opacity ${queue.loading ? "opacity-60" : ""}`} role="region" aria-label="Alert queue" tabIndex={0}>
            <table className="dense" aria-label="Ranked alert queue">
              <thead><tr><th>#</th><th>vs p-only</th><th>Txn</th><th>Cluster</th><th>Amount</th><th>Risk p</th><th>Anomaly a</th><th>Stability e</th><th>Score R</th><th>Top drivers</th><th>Verdict</th></tr></thead>
              <tbody key={version}>
                {q.alerts.map((a, i) => <Row key={a.alert_id} a={a} i={i} selected={i === sel} baseRank={baseRank.get(a.txn_id)} onSelect={() => setSel(i)} />)}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-2 text-[11px] text-soft">Keyboard: <kbd className="chip">j</kbd> <kbd className="chip">k</kbd> move, <kbd className="chip">Enter</kbd> open evidence. <Counter value={q?.alerts.length ?? 0} /> alerts shown.</p>
      </Card>
    </div>
  );
}

function Row({ a, i, selected, baseRank, onSelect }: { a: AlertRow; i: number; selected: boolean; baseRank?: number; onSelect: () => void }) {
  const delta = baseRank == null ? null : baseRank - a.queue_rank;
  return (
    <tr id={`row-${i}`} className={`row-in ${selected ? "sel" : ""}`} style={{ animationDelay: `${Math.min(i, 24) * 22}ms` }} aria-selected={selected}
      onClick={() => { onSelect(); go(`#/alert/${a.alert_id}`); }}>
      <td className="num text-soft">{a.queue_rank}</td>
      <td className="num text-[11px]">
        {delta == null ? <span className="text-cyan" title="not in the p-only queue">{"★"} new</span>
          : delta === 0 ? <span className="text-soft">{"–"}</span>
          : delta > 0 ? <span className="text-lo">{"▲"} {delta}</span> : <span className="text-hi">{"▼"} {-delta}</span>}
      </td>
      <td className="num">#{a.txn_id}</td>
      <td><ClusterTag id={a.cluster_key} size={a.cluster_size} /></td>
      <td className="num">{money(a.amount)}</td>
      <td><RiskBar p={a.p} /></td>
      <td><MiniBar v={a.a} /></td>
      <td><StabilityBar e={a.e} /></td>
      <td className="num">{a.score.toFixed(3)}</td>
      <td className="max-w-[220px] truncate text-soft">{a.top_features.map(([f]) => f).join(", ")}</td>
      <td>{a.verdict ? <span className="chip border-violet/50 text-violet">{a.verdict}</span> : <span className="text-soft">open</span>}</td>
    </tr>
  );
}
