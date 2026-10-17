import { useEffect, useRef, useState } from "react";
import { type AuditPage as AuditData, api, qs } from "../api";
import { Card, Empty, ErrorBox, Skeleton } from "../components/ui";
import { go, useAsync } from "../lib";

const ACTIONS = ["", "view_alert", "decision", "load_run"];

export default function AuditPage() {
  const [page, setPage] = useState(1);
  const [action, setAction] = useState("");
  const [auto, setAuto] = useState(true);
  const audit = useAsync(() => api<AuditData>(`/api/audit${qs({ page, size: 15, action })}`), [page, action]);
  const reload = audit.reload;
  useEffect(() => { if (!auto) return; const t = setInterval(reload, 8000); return () => clearInterval(t); }, [auto, reload]);
  const seen = useRef<Set<number>>(new Set());
  const d = audit.data;
  const fresh = new Set<number>();
  d?.items.forEach((i) => { if (seen.current.size && !seen.current.has(i.id)) fresh.add(i.id); });
  useEffect(() => { d?.items.forEach((i) => seen.current.add(i.id)); });
  const pages = d ? Math.max(1, Math.ceil(d.total / d.size)) : 1;

  return (
    <div className="page-in grid gap-4">
      <Card title="Audit trail" right={<>
        <select className="input" aria-label="Filter action" value={action} onChange={(e) => { setAction(e.target.value); setPage(1); }}>{ACTIONS.map((a) => <option key={a} value={a}>{a || "all actions"}</option>)}</select>
        <label className="flex items-center gap-1.5 text-[12px] text-soft"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> auto-refresh</label>
        <a className="btn" href="/api/audit/export.csv" download>Export CSV</a></>}>
        {audit.error && <ErrorBox error={audit.error} retry={audit.reload} />}
        {!d && !audit.error && <Skeleton h={300} />}
        {d && d.items.length === 0 && <Empty title="No audit entries yet" hint="Opening an alert or recording a decision writes an entry." />}
        {d && d.items.length > 0 && (
          <div className="overflow-x-auto">
            <table className="dense" aria-label="Audit trail"><thead><tr><th>#</th><th>When</th><th>Who</th><th>Action</th><th>Detail</th></tr></thead>
              <tbody>{d.items.map((i, n) => (
                <tr key={i.id} className={`row-in ${fresh.has(i.id) ? "sel" : ""}`} style={{ animationDelay: `${n * 25}ms` }}
                  onClick={() => typeof i.detail?.alert_id === "number" && go(`#/alert/${i.detail.alert_id}`)}>
                  <td className="num text-soft">{i.id}</td><td className="num">{i.at.slice(0, 19).replace("T", " ")}</td><td>{i.actor}</td>
                  <td><span className={`chip ${i.action === "decision" ? "border-violet/60 text-violet" : ""}`}>{i.action}</span></td>
                  <td className="max-w-[420px] truncate num text-soft" title={JSON.stringify(i.detail)}>{JSON.stringify(i.detail)}</td></tr>))}</tbody></table>
          </div>
        )}
        <div className="mt-3 flex items-center justify-between text-[12px] text-soft">
          <span>{d?.total ?? 0} entries</span>
          <div className="flex items-center gap-2">
            <button className="btn" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>Prev</button>
            <span className="num">{page} / {pages}</span>
            <button className="btn" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>Next</button>
          </div>
        </div>
      </Card>
    </div>
  );
}
