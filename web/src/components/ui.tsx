import { type ReactNode, useEffect, useRef, useState } from "react";
import { clamp, hue, riskLevel, stabilityLevel, useCountUp } from "../lib";

export function Card({ title, right, children, className = "", hover = false }: { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; hover?: boolean }) {
  return (
    <section className={`glass ${hover ? "hover" : ""} p-4 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-[13px] font-semibold tracking-wide text-text/90">{title}</h2>
          <div className="flex items-center gap-2">{right}</div>
        </header>
      )}
      {children}
    </section>
  );
}

export function Counter({ value, digits = 0, suffix = "", className = "" }: { value: number; digits?: number; suffix?: string; className?: string }) {
  const v = useCountUp(Number.isFinite(value) ? value : 0);
  return <span className={`num ${className}`} data-testid="counter">{v.toFixed(digits)}{suffix}</span>;
}

export function Stat({ label, value, digits = 0, suffix = "", hint, tone = "text-text" }: { label: string; value: number; digits?: number; suffix?: string; hint?: ReactNode; tone?: string }) {
  return (
    <div className="glass hover relative overflow-hidden px-4 py-3">
      <div className="text-[11px] uppercase tracking-[.08em] text-soft">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${tone}`}><Counter value={value} digits={digits} suffix={suffix} /></div>
      {hint && <div className="mt-0.5 text-[11px] text-soft">{hint}</div>}
    </div>
  );
}

/** Segmented control with a sliding thumb. */
export function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  return (
    <div role="radiogroup" aria-label={label} className="relative inline-grid rounded-xl border border-line bg-[#0a1020] p-0.5" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
      <span className="absolute inset-y-0.5 rounded-[10px] bg-gradient-to-r from-violet/80 to-cyan/70 transition-transform duration-300"
        style={{ width: `calc(${100 / options.length}% - 2px)`, left: 2, transform: `translateX(${idx * 100}%)`, transitionTimingFunction: "var(--ease)" }} />
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={o.value === value} onClick={() => onChange(o.value)}
          className={`relative z-10 whitespace-nowrap px-3 py-1.5 text-[12px] transition-colors ${o.value === value ? "font-semibold text-[#050816]" : "text-soft hover:text-text"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({ label, value, min, max, step = 1, onChange, format, disabled, hint }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; format?: (v: number) => string; disabled?: boolean; hint?: string }) {
  const id = `sl-${label.replace(/\W+/g, "-")}`;
  return (
    <div className={disabled ? "opacity-40" : ""}>
      <div className="mb-1 flex items-baseline justify-between text-[12px]">
        <label htmlFor={id} className="text-soft">{label}</label>
        <span className="num text-text">{format ? format(value) : value}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
      {hint && <div className="mt-0.5 text-[10.5px] text-soft/80">{hint}</div>}
    </div>
  );
}

/** Risk is colour AND a numeric bar; stability is a separate glyph + word + bar (never colour alone). */
export function RiskBar({ p }: { p: number }) {
  const lvl = riskLevel(p), col = { lo: "bg-lo", mid: "bg-mid", hi: "bg-hi" }[lvl];
  return (
    <div className="flex items-center gap-2" title={`fraud probability ${(p * 100).toFixed(1)}%`}>
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-line"><div className={`fill h-full ${col} ${lvl === "hi" ? "risk-glow-hi" : ""}`} style={{ width: `${clamp(p, 0.02, 1) * 100}%` }} /></div>
      <span className="num w-12 text-right">{(p * 100).toFixed(1)}%</span>
    </div>
  );
}
export function MiniBar({ v, tone = "bg-violet" }: { v: number; tone?: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-line"><div className={`fill h-full ${tone}`} style={{ width: `${clamp(v, 0.02, 1) * 100}%` }} /></div>
      <span className="num w-9 text-right text-soft">{v.toFixed(2)}</span>
    </div>
  );
}
export function StabilityBar({ e, wide = false }: { e: number; wide?: boolean }) {
  const s = stabilityLevel(e);
  return (
    <div className="flex items-center gap-2" title={`explanation stability ${e.toFixed(2)} (${s.label})`}>
      <span aria-hidden className="w-3 text-cyan">{s.glyph}</span>
      <div className={`${wide ? "w-40" : "w-16"} h-1.5 overflow-hidden rounded-full border border-cyan/30 bg-transparent`}>
        <div className="fill h-full bg-cyan/80" style={{ width: `${clamp(e, 0.03, 1) * 100}%`, backgroundImage: "repeating-linear-gradient(90deg,#22d3ee 0 6px,#22d3eecc 6px 8px)" }} />
      </div>
      <span className="num text-soft">{e.toFixed(2)}</span>
      <span className="text-[10.5px] text-soft">{s.label}</span>
    </div>
  );
}

export function ClusterTag({ id, size }: { id: string; size: number }) {
  const h = hue(id);
  return (
    <span className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px]" title={`cluster ${id}`}
      style={{ background: `hsl(${h} 60% 30% / .45)`, border: `1px solid hsl(${h} 70% 55% / .5)` }}>
      <span aria-hidden>{"●"}</span><span className="num max-w-[84px] truncate">{id}</span>
      {size > 1 && <b className="rounded bg-black/30 px-1">{"×"}{size}</b>}
    </span>
  );
}

export function Skeleton({ h = 120, className = "" }: { h?: number; className?: string }) {
  return <div className={`shimmer ${className}`} style={{ height: h }} aria-busy="true" aria-label="Loading" />;
}
export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="grid place-items-center gap-1 py-10 text-center" role="status">
      <div className="text-3xl opacity-50" aria-hidden>{"◌"}</div>
      <div className="text-sm font-medium">{title}</div>
      {hint && <div className="max-w-sm text-xs text-soft">{hint}</div>}
    </div>
  );
}
export function ErrorBox({ error, retry }: { error: Error; retry?: () => void }) {
  return (
    <div role="alert" className="glass border-hi/40 p-4 text-sm">
      <div className="font-medium text-hi">Something went wrong</div>
      <div className="mt-1 text-xs text-soft">{error.message}</div>
      {retry && <button className="btn mt-3" onClick={retry}>Retry</button>}
    </div>
  );
}

/* ---------- toasts ---------- */
type T = { id: number; msg: string; kind: "ok" | "err" | "info" };
const listeners = new Set<(t: T) => void>();
let seq = 0;
export const toast = (msg: string, kind: T["kind"] = "info") => listeners.forEach((l) => l({ id: ++seq, msg, kind }));
export function Toasts() {
  const [items, setItems] = useState<T[]>([]);
  useEffect(() => {
    const l = (t: T) => { setItems((x) => [...x, t]); setTimeout(() => setItems((x) => x.filter((i) => i.id !== t.id)), 3800); };
    listeners.add(l);
    return () => { listeners.delete(l); };
  }, []);
  return (
    <div className="fixed right-4 top-4 z-50 flex flex-col gap-2" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className={`toast glass max-w-xs px-4 py-2.5 text-[13px] ${t.kind === "ok" ? "border-lo/50" : t.kind === "err" ? "border-hi/60" : ""}`}>
          <span className="mr-2" aria-hidden>{t.kind === "ok" ? "✓" : t.kind === "err" ? "✕" : "ℹ"}</span>{t.msg}
        </div>
      ))}
    </div>
  );
}

/** Radial risk gauge with animated sweep. */
export function Gauge({ p }: { p: number }) {
  const r = 52, c = 2 * Math.PI * r, lvl = riskLevel(p);
  const col = { lo: "#34d399", mid: "#fbbf24", hi: "#f43f5e" }[lvl];
  const [on, setOn] = useState(false);
  useEffect(() => { const t = setTimeout(() => setOn(true), 60); return () => clearTimeout(t); }, []);
  return (
    <div className="relative grid h-36 w-36 place-items-center" role="img" aria-label={`Fraud probability ${(p * 100).toFixed(1)} percent, ${lvl} risk`}>
      <svg viewBox="0 0 130 130" className="absolute inset-0 -rotate-90">
        <circle cx="65" cy="65" r={r} fill="none" stroke="#1f2a47" strokeWidth="9" />
        <circle cx="65" cy="65" r={r} fill="none" stroke={col} strokeWidth="9" strokeLinecap="round" strokeDasharray={c}
          strokeDashoffset={on ? c * (1 - clamp(p, 0.01, 1)) : c} style={{ transition: "stroke-dashoffset 1.2s var(--ease)", filter: `drop-shadow(0 0 8px ${col}99)` }} />
      </svg>
      <div className="text-center"><div className="num text-2xl font-semibold"><Counter value={p * 100} digits={1} suffix="%" /></div><div className="text-[10px] uppercase tracking-widest text-soft">{lvl} risk</div></div>
    </div>
  );
}

export function useOnKey(handler: (e: KeyboardEvent) => void) {
  const ref = useRef(handler);
  ref.current = handler;
  useEffect(() => {
    const f = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && /input|textarea|select/i.test(t.tagName) && !(e.ctrlKey || e.metaKey)) return;
      ref.current(e);
    };
    window.addEventListener("keydown", f);
    return () => window.removeEventListener("keydown", f);
  }, []);
}
