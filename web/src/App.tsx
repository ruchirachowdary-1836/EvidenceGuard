import { useEffect, useMemo, useRef, useState } from "react";
import { type Meta, analyst, api, setAnalyst } from "./api";
import { ErrorBox, Skeleton, Toasts, useOnKey } from "./components/ui";
import { go, useAsync, useHashRoute } from "./lib";
import { MetaContext } from "./meta";
import AuditPage from "./pages/AuditPage";
import DetailPage from "./pages/DetailPage";
import ModelsPage from "./pages/ModelsPage";
import QueuePage from "./pages/QueuePage";
import SimulatorPage from "./pages/SimulatorPage";

const NAV = [
  { path: "#/queue", label: "Queue", key: "1", icon: "M4 6h16M4 12h16M4 18h10" },
  { path: "#/models", label: "Models", key: "2", icon: "M3 3v18h18M7 15l4-5 3 3 5-7" },
  { path: "#/simulator", label: "Simulator", key: "3", icon: "M12 3v4M12 17v4M3 12h4M17 12h4M12 8a4 4 0 100 8 4 4 0 000-8z" },
  { path: "#/audit", label: "Audit", key: "4", icon: "M9 12l2 2 4-4M12 3l8 3v6c0 5-4 8-8 9-4-1-8-4-8-9V6z" },
];

function Palette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [q, setQ] = useState("");
  const [i, setI] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const items = useMemo(() => [
    ...NAV.map((n) => ({ label: `Go to ${n.label}`, run: () => go(n.path) })),
    { label: "Export audit trail (CSV)", run: () => { window.location.href = "/api/audit/export.csv"; } },
  ].filter((x) => x.label.toLowerCase().includes(q.toLowerCase())), [q]);
  useEffect(() => { if (open) { setQ(""); setI(0); setTimeout(() => input.current?.focus(), 30); } }, [open]);
  if (!open) return null;
  const choose = (n: number) => { items[n]?.run(); onClose(); };
  return (
    <div className="fixed inset-0 z-50 grid place-items-start justify-center bg-black/60 pt-[18vh] backdrop-blur-sm" onClick={onClose} role="dialog" aria-modal="true" aria-label="Command palette">
      <div className="glass page-in w-[min(520px,92vw)] overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <input ref={input} className="w-full border-b border-line bg-transparent px-4 py-3.5 text-sm outline-none" placeholder="Jump to..." value={q} aria-label="Command"
          onChange={(e) => { setQ(e.target.value); setI(0); }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setI((x) => Math.min(items.length - 1, x + 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setI((x) => Math.max(0, x - 1)); }
            if (e.key === "Enter") choose(i);
            if (e.key === "Escape") onClose();
          }} />
        <ul className="max-h-64 overflow-auto p-1.5">
          {items.map((x, n) => <li key={x.label}><button className={`w-full rounded-lg px-3 py-2 text-left text-[13px] ${n === i ? "bg-violet/30" : "hover:bg-panel2"}`} onMouseEnter={() => setI(n)} onClick={() => choose(n)}>{x.label}</button></li>)}
          {!items.length && <li className="px-3 py-3 text-xs text-soft">No matches</li>}
        </ul>
      </div>
    </div>
  );
}

export default function App() {
  const route = useHashRoute();
  const meta = useAsync(() => api<Meta>("/api/meta"), []);
  const [palette, setPalette] = useState(false);
  const [who, setWho] = useState(analyst());
  useOnKey((e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); setPalette((p) => !p); return; }
    const n = NAV.find((x) => x.key === e.key);
    if (n && !e.ctrlKey && !e.metaKey && !e.altKey) go(n.path);
  });

  let page;
  const m = route.match(/^#\/alert\/(\d+)/);
  if (m) page = <DetailPage id={Number(m[1])} />;
  else if (route.startsWith("#/models")) page = <ModelsPage />;
  else if (route.startsWith("#/simulator")) page = <SimulatorPage />;
  else if (route.startsWith("#/audit")) page = <AuditPage />;
  else page = <QueuePage />;
  const active = m ? "#/queue" : NAV.find((n) => route.startsWith(n.path))?.path ?? "#/queue";

  return (
    <div className="relative min-h-full">
      <div className="aurora" aria-hidden><i /><i /><i /></div>
      <Toasts />
      <Palette open={palette} onClose={() => setPalette(false)} />
      <div className="relative z-10 flex min-h-screen">
        <nav className="sticky top-0 hidden h-screen w-[84px] shrink-0 flex-col items-center gap-1.5 border-r border-line bg-panel/70 py-4 backdrop-blur md:flex" aria-label="Main">
          <div className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-violet to-cyan text-[#050816] shadow-[0_8px_30px_-8px_#8b5cf6]" title="EvidenceGuard">
            <svg viewBox="0 0 32 32" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinejoin="round" strokeLinecap="round"><path d="M16 3l11 4v8c0 7-5 12-11 14C10 27 5 22 5 15V7z" /><path d="M11 16l4 4 7-8" /></svg>
          </div>
          {NAV.map((n) => (
            <a key={n.path} href={n.path} aria-current={active === n.path ? "page" : undefined} title={`${n.label} (${n.key})`}
              className={`flex w-[64px] flex-col items-center gap-1 rounded-xl py-2 text-[10.5px] transition ${active === n.path ? "bg-gradient-to-br from-violet/90 to-cyan/80 font-semibold text-[#050816]" : "text-soft hover:bg-panel2 hover:text-text"}`}>
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round"><path d={n.icon} /></svg>{n.label}
            </a>
          ))}
        </nav>
        <div className="flex min-w-0 flex-1 flex-col">
          <header className="sticky top-0 z-20 flex flex-wrap items-center gap-3 border-b border-line bg-ink/70 px-5 py-3 backdrop-blur">
            <div>
              <div className="text-[15px] font-semibold tracking-tight">Evidence<span className="bg-gradient-to-r from-violet to-cyan bg-clip-text text-transparent">Guard</span> <span className="text-soft">analyst console</span></div>
              <div className="text-[11px] text-soft">Explanation-aware, budgeted fraud alert ranking</div>
            </div>
            <div className="ml-auto flex flex-wrap items-center gap-2">
              {meta.data && <span className="chip" title="Model version">{"◆"} {meta.data.model_version}</span>}
              <button className="btn" onClick={() => setPalette(true)} aria-label="Open command palette">Search <kbd className="chip ml-1">Ctrl K</kbd></button>
              <label className="flex items-center gap-1.5 text-[12px] text-soft">Analyst
                <input className="input w-28" value={who} aria-label="Analyst name" onChange={(e) => { setWho(e.target.value); setAnalyst(e.target.value); }} maxLength={40} /></label>
            </div>
          </header>
          <nav className="flex gap-1 overflow-x-auto border-b border-line px-3 py-1.5 md:hidden" aria-label="Mobile">
            {NAV.map((n) => <a key={n.path} href={n.path} className={`rounded-lg px-3 py-1.5 text-[12px] ${active === n.path ? "bg-violet/40 font-semibold" : "text-soft"}`}>{n.label}</a>)}
          </nav>
          <main className="flex-1 px-5 py-5">
            {meta.error && <ErrorBox error={meta.error} retry={meta.reload} />}
            {!meta.data && !meta.error && <div className="grid gap-4"><Skeleton h={90} /><Skeleton h={200} /><Skeleton h={320} /></div>}
            {meta.data && <MetaContext.Provider value={meta.data}>{page}</MetaContext.Provider>}
          </main>
          <footer className="border-t border-line bg-panel/60 px-5 py-2.5 text-[11.5px] text-soft backdrop-blur" role="contentinfo">
            <span className="mr-2 rounded bg-mid/15 px-1.5 py-0.5 font-medium text-mid">Dataset caveat</span>
            {meta.data?.caveat ?? "Research prototype. Not real Indian UPI traffic; no claim of UPI effectiveness."}
            {meta.data && <span className="ml-3 num">model {meta.data.model_version} | run {meta.data.run_id}</span>}
          </footer>
        </div>
      </div>
    </div>
  );
}
