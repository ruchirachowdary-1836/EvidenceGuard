import { useCallback, useEffect, useRef, useState } from "react";

export const pct = (v: number | null | undefined, d = 1) => (v == null || !Number.isFinite(v) ? "-" : `${(v * 100).toFixed(d)}%`);
export const f3 = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "-" : v.toFixed(3));
export const money = (v: number) => v.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function hue(s: string): number {
  let h = 0;
  for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return h;
}

/** Calendar anchor of the synthetic time axis (day 0). Matches ml.run_experiment.EPOCH. */
export function clock(ts: number): string {
  const h = Math.floor((ts % 86400) / 3600), m = Math.floor((ts % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export function riskLevel(p: number): "lo" | "mid" | "hi" {
  return p >= 0.5 ? "hi" : p >= 0.15 ? "mid" : "lo";
}
export function stabilityLevel(e: number): { label: "stable" | "shaky" | "unstable"; glyph: string } {
  return e >= 0.8 ? { label: "stable", glyph: "◆" } : e >= 0.55 ? { label: "shaky", glyph: "▲" } : { label: "unstable", glyph: "▼" };
}

export function useDebounced<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
}

export interface Async<T> { data: T | undefined; error: Error | undefined; loading: boolean; reload: () => void }

/** Fetch with stale-response protection; keeps the previous data visible while reloading (smooth transitions). */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): Async<T> {
  const [state, setState] = useState<{ data?: T; error?: Error; loading: boolean }>({ loading: true });
  const seq = useRef(0);
  const [tick, setTick] = useState(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(fn, deps);
  useEffect(() => {
    const id = ++seq.current;
    setState((s) => ({ ...s, loading: true }));
    run().then(
      (data) => { if (id === seq.current) setState({ data, loading: false }); },
      (error: Error) => { if (id === seq.current) setState((s) => ({ ...s, error, loading: false })); },
    );
  }, [run, tick]);
  return { data: state.data, error: state.error, loading: state.loading, reload: () => setTick((t) => t + 1) };
}

/** Animated number: eases from the previous value to the new one. */
export function useCountUp(target: number, ms = 700): number {
  const [v, setV] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = performance.now(), a = from.current;
    if (a === target) return;
    let raf = 0;
    const step = (now: number) => {
      const t = clamp((now - start) / ms, 0, 1), e = 1 - Math.pow(1 - t, 3);
      const cur = a + (target - a) * e;
      setV(cur); from.current = cur;
      if (t < 1) raf = requestAnimationFrame(step);
      else from.current = target;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

export function useHashRoute(): string {
  const [h, setH] = useState(() => window.location.hash || "#/queue");
  useEffect(() => {
    const f = () => setH(window.location.hash || "#/queue");
    window.addEventListener("hashchange", f);
    return () => window.removeEventListener("hashchange", f);
  }, []);
  return h;
}
export const go = (path: string) => { window.location.hash = path; };
