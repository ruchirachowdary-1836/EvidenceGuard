import json

import pytest


def test_health_and_meta(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    m = client.get("/api/meta").json()
    assert m["run_id"] == "pytest-run" and m["days"] and m["default_day"] in m["days"] and "synthetic" in m["caveat"].lower()
    assert "UPI" in m["caveat"]


def test_queue_is_budgeted_and_ranked(client):
    q = client.get("/api/alerts", params={"method": "full", "k": 8}).json()
    assert q["k"] == 8 and len(q["alerts"]) == 8
    scores = [a["score"] for a in q["alerts"]]
    assert [a["queue_rank"] for a in q["alerts"]] == list(range(1, 9))
    assert all(0 <= a["p"] <= 1 and 0 <= a["e"] <= 1 for a in q["alerts"])
    assert len({a["txn_id"] for a in q["alerts"]}) == 8
    assert all(len(a["top_features"]) <= 3 for a in q["alerts"])
    assert scores[0] >= 0


def test_methods_and_lambda_change_the_queue(client):
    base = client.get("/api/alerts", params={"method": "p_only", "k": 15}).json()
    p = [a["p"] for a in base["alerts"]]
    assert p == sorted(p, reverse=True)
    nolam = client.get("/api/alerts", params={"method": "full", "k": 15, "lam": 0}).json()
    allm = client.get("/api/alerts", params={"method": "all", "k": 15}).json()
    # with lambda = 0 the full method is the same queue as the all-components method (same weights)
    assert [a["txn_id"] for a in nolam["alerts"]] == [a["txn_id"] for a in allm["alerts"]] or nolam["weights"] != allm["weights"]
    strong = client.get("/api/alerts", params={"method": "full", "k": 15, "lam": 0.9}).json()
    assert strong["redundancy"] <= nolam["redundancy"] + 1e-9


def test_validation_errors(client):
    assert client.get("/api/alerts", params={"method": "bogus"}).status_code == 422
    assert client.get("/api/alerts", params={"day": "1999-01-01"}).status_code == 404
    assert client.get("/api/alerts", params={"lam": 5}).status_code == 422


def test_alert_detail_hides_label_and_writes_audit(client):
    first = client.get("/api/alerts", params={"k": 3}).json()["alerts"][0]
    d = client.get(f"/api/alerts/{first['alert_id']}", headers={"X-Analyst": "asha"}).json()
    assert d["transaction"]["label"] is None                      # evaluation-only field is not exposed by default
    assert d["shap"] and d["shap"][0]["rank"] == 1 and len(d["stability"]) == 6
    assert d["baseline"]["n"] >= 0 and "median" in d["baseline"]
    assert isinstance(d["peers"], list)
    assert client.get(f"/api/alerts/{first['alert_id']}", params={"reveal": True}).json()["transaction"]["label"] in (0, 1)
    audit = client.get("/api/audit", params={"action": "view_alert"}).json()
    assert audit["total"] >= 2 and audit["items"][-1]["actor"] in ("asha", "analyst")
    assert client.get("/api/alerts/99999999").status_code == 404


def test_decision_writes_decision_and_audit(client):
    a = client.get("/api/alerts", params={"k": 2}).json()["alerts"][1]
    r = client.post(f"/api/alerts/{a['alert_id']}/decision", json={"analyst": "ravi", "verdict": "fraud", "note": "ring device"})
    assert r.status_code == 201
    detail = client.get(f"/api/alerts/{a['alert_id']}").json()
    assert detail["decisions"][0]["verdict"] == "fraud"
    queue = client.get("/api/alerts", params={"k": 2}).json()["alerts"]
    assert any(x["verdict"] == "fraud" for x in queue)
    rows = client.get("/api/audit", params={"action": "decision"}).json()["items"]
    assert rows[0]["detail"]["alert_id"] == a["alert_id"] and rows[0]["actor"] == "ravi"
    assert client.post(f"/api/alerts/{a['alert_id']}/decision", json={"verdict": "maybe"}).status_code == 422
    assert client.post("/api/alerts/99999999/decision", json={"verdict": "fraud"}).status_code == 404


def test_simulate_returns_metrics_and_baseline(client):
    r = client.post("/api/simulate", json={"method": "all", "k": 10, "beta": 0.5, "gamma": 0.5}).json()
    assert 0 <= r["recall"] <= 1 and 0 <= r["precision"] <= 1
    assert r["queued"] <= 10 * r["days"] and r["curve"] and r["per_day"]
    assert r["delta_recall"] == pytest.approx(r["recall"] - r["baseline"]["recall"])
    same = client.post("/api/simulate", json={"method": "p_only", "k": 10}).json()
    assert same["recall"] == pytest.approx(same["baseline"]["recall"])
    lam0 = client.post("/api/simulate", json={"method": "full", "k": 10, "lam": 0}).json()
    allm = client.post("/api/simulate", json={"method": "all", "k": 10}).json()
    assert lam0["recall"] == pytest.approx(allm["recall"])        # lambda = 0 reduces to plain top-K
    assert client.post("/api/simulate", json={"method": "all", "k": 0}).status_code == 422


def test_experiments_and_curves(client):
    runs = client.get("/api/experiments").json()
    assert runs[0]["run_id"] == "pytest-run" and runs[0]["headline"]["pr_auc_xgb"] is not None
    c = client.get("/api/experiments/pytest-run/curves").json()
    variants = {r["variant"] for r in c["ablation"]}
    assert variants == {"p_only", "p_a", "p_e", "all", "full"}
    assert all(0 <= r["recall_lo"] <= r["recall"] <= r["recall_hi"] <= 1 for r in c["ablation"])
    assert "logreg" in c["baselines"] and c["stability"]["hist_fraud"]
    assert client.get("/api/experiments/nope/curves").status_code == 404


def test_audit_paging_and_csv_injection_safe(client):
    a = client.get("/api/alerts", params={"k": 1}).json()["alerts"][0]
    client.post(f"/api/alerts/{a['alert_id']}/decision", json={"analyst": "=cmd()", "verdict": "legit"})
    page = client.get("/api/audit", params={"size": 2}).json()
    assert len(page["items"]) == 2 and page["total"] >= 2
    csv_text = client.get("/api/audit/export.csv").text
    assert csv_text.startswith("id,at,actor,action,detail") and "'=cmd()" in csv_text


def test_stream_replays_candidates(client):
    with client.stream("GET", "/api/stream", params={"interval": 0.02, "limit": 3}) as r:
        body = "".join(r.iter_text())
    events = [json.loads(l[6:]) for l in body.splitlines() if l.startswith("data: ")]
    assert len(events) == 4 and "note" in events[0] and events[1]["alert_id"]
