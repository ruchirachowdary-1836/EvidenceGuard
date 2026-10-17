import numpy as np
import pytest

from ml.stability import StabilityEngine, benign_perturb, deletion_test, explanation_stability, jaccard, topk_set


def test_jaccard_and_topk():
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 1.0
    assert topk_set(np.array([0.1, -5.0, 2.0, 0.0]), 2) == {1, 2}


def test_stable_explainer_scores_one():
    explain = lambda x: np.array([5.0, 4.0, 3.0, 0.1, 0.1, 0.1])        # perturbation-proof ranking
    perturb = lambda x, rng: x + rng.normal(0, 0.01, x.shape)
    assert explanation_stability(np.zeros(6), explain, perturb, k=3, M=20) == pytest.approx(1.0)


def test_unstable_explainer_scores_low():
    explain = lambda x: x                                            # attributions = the (noisy) input itself
    perturb = lambda x, rng: rng.normal(0, 1, x.shape)
    assert explanation_stability(np.zeros(10), explain, perturb, k=3, M=40) < 0.6


def test_benign_perturb_preserves_scenario():
    rng = np.random.default_rng(0)
    pool = np.arange(30, dtype=float).reshape(5, 6)
    x = np.array([100.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    z = benign_perturb(x, rng, 0, [3, 4], pool, amt_tol=0.02)
    assert 98 <= z[0] <= 102 and z[1] == 1.0 and z[2] == 2.0 and z[5] == 5.0


def test_stability_is_deterministic_per_seed():
    a = explanation_stability(np.zeros(8), lambda x: x, lambda x, r: r.normal(0, 1, x.shape), seed=5)
    b = explanation_stability(np.zeros(8), lambda x: x, lambda x, r: r.normal(0, 1, x.shape), seed=5)
    assert a == b


@pytest.fixture(scope="module")
def engine_and_data(tiny_feat):
    from ml.models import Calibrator, fit_space, train_xgb
    n = len(tiny_feat)
    tr, va = tiny_feat.iloc[: int(n * .6)], tiny_feat.iloc[int(n * .6): int(n * .8)]
    sp = fit_space(tr)
    Xtr, Xva = sp.matrix(tr), sp.matrix(va)
    m = train_xgb(Xtr, tr.label.to_numpy(), Xva, va.label.to_numpy(),
                  {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.15, "early_stopping_rounds": 10}, True, 0)
    cal = Calibrator().fit(m.margin(Xva), va.label.to_numpy())
    cfg = {"k": 5, "M": 8, "amt_tol": 0.02, "time_shift_s": 300, "delta": 0.05}
    return StabilityEngine(m, cal, sp.names, Xtr, cfg), Xva[:25], sp


def test_engine_outputs_valid_ranges(engine_and_data):
    eng, X, _ = engine_and_data
    r = eng.run(X, seed=1)
    assert r.e.shape == (25,) and ((r.e >= 0) & (r.e <= 1)).all()
    assert r.jaccard.shape == (25, 8) and r.kept.dtype == bool
    assert (r.topk.sum(axis=2) == 5).all()                          # exactly k features per perturbation
    assert ((r.prob_shift <= 0.05) == r.kept).all()                 # benign filter is the delta rule


def test_engine_deterministic(engine_and_data):
    eng, X, _ = engine_and_data
    assert np.allclose(eng.run(X, seed=4).e, eng.run(X, seed=4).e)


def test_deletion_test_reports_faithfulness(engine_and_data):
    eng, X, sp = engine_and_data
    r = eng.run(X, seed=2)
    out = deletion_test(eng, X, r.shap, r.p, sp.medians[sp.names].to_numpy(np.float32), seed=0)
    assert set(out) == {"drop_topk", "drop_random", "faithful_frac"} and 0 <= out["faithful_frac"] <= 1
