import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import * as fx from "./fixtures";

/** Mocked API: every page is smoke-tested against canned responses (no backend needed). */
function mockApi() {
  const calls: { url: string; init?: RequestInit }[] = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, init });
    const table: [RegExp, unknown][] = [
      [/\/api\/meta/, fx.meta], [/\/api\/alerts\/\d+\/decision/, { alert_id: 101, verdict: "fraud" }], [/\/api\/alerts\/\d+/, fx.detail],
      [/\/api\/alerts\?.*method=p_only/, fx.baseQueue], [/\/api\/alerts/, fx.queue], [/\/api\/simulate/, fx.sim],
      [/\/api\/experiments\/.+\/curves/, fx.curves], [/\/api\/experiments/, fx.experiments], [/\/api\/audit/, fx.audit],
    ];
    for (const [re, body] of table) {
      if (re.test(url)) return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response("{}", { status: 404 });
  }));
  return calls;
}

beforeEach(() => {
  window.location.hash = "";
  vi.stubGlobal("EventSource", class { addEventListener() {} close() {} onerror = null; });
});

describe("EvidenceGuard dashboard", () => {
  it("shows the dataset caveat and model version in the footer", async () => {
    mockApi();
    render(<App />);
    expect(await screen.findByRole("contentinfo")).toHaveTextContent(/not real Indian UPI traffic/i);
    await waitFor(() => expect(screen.getByRole("contentinfo")).toHaveTextContent("xgb-test"));
  });

  it("queue page renders ranked alerts with a stability word next to the colour", async () => {
    mockApi();
    render(<App />);
    expect(await screen.findByRole("table", { name: /ranked alert queue/i })).toBeInTheDocument();
    expect((await screen.findAllByText("#5001")).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/shaky|unstable|stable/).length).toBeGreaterThan(0);
    expect(screen.getAllByText("fraud").length).toBeGreaterThan(0);          // the recorded verdict is shown
  });

  it("switching the method and budget re-queries the API", async () => {
    const calls = mockApi();
    render(<App />);
    await screen.findByRole("table", { name: /ranked alert queue/i });
    await userEvent.click(screen.getByRole("radio", { name: "p + stability" }));
    await waitFor(() => expect(calls.some((c) => c.url.includes("method=p_e"))).toBe(true));
    fireEvent.change(screen.getByLabelText(/Capacity K/i), { target: { value: "9" } });
    await waitFor(() => expect(calls.some((c) => /[?&]k=9(&|$)/.test(c.url))).toBe(true), { timeout: 2000 });
  });

  it("keyboard navigation opens the selected alert", async () => {
    mockApi();
    render(<App />);
    await screen.findByRole("table", { name: /ranked alert queue/i });
    fireEvent.keyDown(window, { key: "j" });
    fireEvent.keyDown(window, { key: "Enter" });
    expect(window.location.hash).toBe("#/alert/102");
  });

  it("alert detail shows SHAP and the perturbation heatmap, and posts a decision", async () => {
    const calls = mockApi();
    window.location.hash = "#/alert/101";
    render(<App />);
    expect(await screen.findByRole("img", { name: /SHAP feature attributions/i })).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /perturbation heatmap/i })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Fraud probability 80\.0 percent/i })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /Confirm fraud/i }));
    await waitFor(() => expect(calls.some((c) => c.url.includes("/decision") && c.init?.method === "POST")).toBe(true));
    const post = calls.find((c) => c.url.includes("/decision"))!;
    expect(JSON.parse(String(post.init?.body))).toMatchObject({ verdict: "fraud" });
  });

  it("models page lists baselines and the ablation table with confidence intervals", async () => {
    mockApi();
    window.location.hash = "#/models";
    render(<App />);
    expect(await screen.findByText(/Ablation: variants 1-5/i)).toBeInTheDocument();
    expect(screen.getAllByText(/95% CI/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/logreg/).length).toBeGreaterThan(0);
    expect(screen.getByRole("status")).toHaveTextContent(/confidence intervals|No variant beat/i);
  });

  it("simulator page shows live metrics against the p-only baseline", async () => {
    const calls = mockApi();
    window.location.hash = "#/simulator";
    render(<App />);
    expect(await screen.findByLabelText("Live metrics")).toBeInTheDocument();
    expect(screen.getAllByText(/pts vs p only/).length).toBeGreaterThan(0);
    fireEvent.change(screen.getByLabelText(/Review capacity K per day/i), { target: { value: "55" } });
    await waitFor(() => expect(calls.some((c) => c.url.includes("/api/simulate") && String(c.init?.body).includes('"k":55'))).toBe(true), { timeout: 2000 });
  });

  it("audit page shows entries and offers CSV export", async () => {
    mockApi();
    window.location.hash = "#/audit";
    render(<App />);
    const table = await screen.findByRole("table", { name: /audit trail/i });
    expect(within(table).getByText("view_alert")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Export CSV/i })).toHaveAttribute("href", "/api/audit/export.csv");
  });

  it("shows an error state with retry when the API fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "No experiment run loaded." }), { status: 404 })));
    render(<App />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/No experiment run loaded/);
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("command palette navigates with Ctrl+K", async () => {
    mockApi();
    render(<App />);
    await screen.findByRole("table", { name: /ranked alert queue/i });
    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const input = await screen.findByLabelText("Command");
    await userEvent.type(input, "audit{Enter}");
    expect(window.location.hash).toBe("#/audit");
  });
});
