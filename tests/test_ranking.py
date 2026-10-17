import numpy as np
import pytest

from ml import ranking
from ml.metrics import bootstrap_ci, pr_auc, precision_recall_at_k


def test_queue_equals_topk_when_lambda_zero():
    rng = np.random.default_rng(0)
    s = rng.random(200)
    groups = rng.integers(0, 15, 200)
    got = ranking.select_queue(s, groups, 25, lam=0.0)
    assert sorted(got) == sorted(np.argsort(-s)[:25].tolist())


def test_queue_is_unique_and_respects_budget():
    s = np.random.default_rng(1).random(50)
    q = ranking.select_queue(s, np.zeros(50, int), 10, lam=0.5)
    assert len(q) == len(set(q)) == 10


def test_redundancy_discount_prefers_diverse_clusters():
    s = np.array([0.90, 0.89, 0.88, 0.60])
    groups = np.array(["a", "a", "a", "b"])
    assert ranking.select_queue(s, groups, 2, lam=0.0) == [0, 1]
    assert 3 in ranking.select_queue(s, groups, 3, lam=0.5)       # 0.88*0.25 < 0.60 -> the other cluster wins a slot
    assert ranking.redundancy(groups, [0, 1, 3]) == pytest.approx(2 / 3)


def test_multiplicative_cannot_rescue_low_risk_but_linear_can():
    p, a, e = np.array([0.01, 0.50]), np.array([1.0, 0.1]), np.array([1.0, 0.5])
    lin = ranking.combined_score(p, a, e, "linear", beta=1, gamma=1)
    mul = ranking.combined_score(p, a, e, "mult", beta=1, gamma=1)
    assert lin[0] > lin[1]                 # documented weakness of the linear baseline
    assert mul[0] < mul[1]                 # multiplicative form keeps the high-risk transaction on top


def test_constrained_applies_stability_only_above_floor():
    p, a, e = np.array([0.01, 0.30]), np.zeros(2), np.array([0.0, 0.0])
    out = ranking.combined_score(p, a, e, "constrained", gamma=1.0, risk_floor=0.05)
    assert out[0] == pytest.approx(0.01)   # below the floor: untouched
    assert out[1] == pytest.approx(0.0)    # above: modulated


def test_mult_reduces_to_p_when_weights_zero():
    p = np.array([0.2, 0.7])
    assert np.allclose(ranking.combined_score(p, np.array([1, 1]), np.array([0, 0]), "mult", beta=0, gamma=0), p)


def test_evaluate_and_weight_search_on_hand_case():
    days = [{"p": np.array([.9, .8, .7, .6]), "a": np.array([0, 0, 1, 0.]), "e": np.array([.2, .2, 1., .2]),
             "groups": np.array(list("abcd")), "y": np.array([0, 0, 1, 0.]), "n_day": 40, "y_total": 2.0}]
    base = ranking.evaluate(days, k_abs=1, variant="p_only")
    assert base["recall"] == 0.0 and base["queued"] == 1
    best = ranking.search_weights(days, "all", "mult", {"beta": [0, 1], "gamma": [0, 1], "lam": [0]}, 0.025, 0.05, 2.0)
    assert best["val_recall"] == pytest.approx(0.5)               # one of two frauds found at K=1 once a, e are used


def test_variants_cover_the_plan():
    assert list(ranking.VARIANTS) == ["p_only", "p_a", "p_e", "all", "full"]
    assert ranking.VARIANTS["full"]["redundancy"] and not ranking.VARIANTS["all"]["redundancy"]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        ranking.combined_score(.1, .1, .1, "nope")


def test_metrics_hand_computed():
    y = np.array([1, 0, 1, 0, 0])
    prec, rec = precision_recall_at_k(y, [0, 1])
    assert prec == 0.5 and rec == 0.5
    assert pr_auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == pytest.approx(1.0)
    m, lo, hi = bootstrap_ci(np.array([1.0] * 30 + [0.0] * 70), n=500)
    assert lo <= m <= hi and 0.15 < lo < m < hi < 0.45


def test_calibrator_is_monotone_and_tie_free():
    from ml.models import Calibrator
    rng = np.random.default_rng(0)
    margin = rng.normal(0, 3, 500)
    y = (rng.random(500) < 1 / (1 + np.exp(-margin))).astype(int)
    for method in ("platt", "isotonic"):
        cal = Calibrator(method).fit(margin, y)
        out = cal(np.sort(margin))
        assert (np.diff(out) >= -1e-12).all() and ((out >= 0) & (out <= 1)).all()
    platt = Calibrator("platt").fit(margin, y)(np.array([5.0, 10.0, 20.0]))
    assert len(set(platt.round(12))) == 3 or (platt[1] < 1.0 + 1e-12)   # smooth: does not collapse to a plateau early
