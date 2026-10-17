"""Leakage tests: past-only features, chronological splits, train-only fitting, shuffled-time collapse."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from ml.data import assert_chronological, chronological_folds
from ml.features import build_features, numeric_columns
from ml.models import TargetEncoder, fit_space, train_xgb

WINDOWS = [3600, 86400, 604800, 2592000]


def test_features_use_only_strictly_earlier_rows(tiny_df, tiny_feat):
    """Recompute on a dataset cut at time T: every row before T must be identical (no future leakage)."""
    cut = int(len(tiny_df) * 0.6)
    t_cut = tiny_df["ts"].iat[cut]
    trunc = tiny_df[tiny_df["ts"] < t_cut].reset_index(drop=True)
    again = build_features(trunc, WINDOWS, 50)
    cols = numeric_columns(tiny_feat)
    full = tiny_feat[tiny_feat["ts"] < t_cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(full[cols], again[cols], check_exact=False, rtol=1e-9, atol=1e-9)


def test_changing_a_future_row_does_not_change_past_features(tiny_df, tiny_feat):
    mod = tiny_df.copy()
    i = len(mod) - 1
    mod.loc[i, ["amount", "label"]] = [999999.0, 1 - mod.loc[i, "label"]]
    again = build_features(mod, WINDOWS, 50)
    cols = numeric_columns(tiny_feat)
    pd.testing.assert_frame_equal(tiny_feat.loc[: i - 1, cols], again.loc[: i - 1, cols], check_exact=False, rtol=1e-9, atol=1e-9)


def test_equal_timestamps_do_not_see_each_other():
    df = pd.DataFrame({"txn_id": [1, 2, 3], "ts": [1000, 1000, 2000], "amount": [10.0, 500.0, 20.0], "product": "W", "card_type": "c",
                       "email_domain": "x", "device_type": "m", "d1": 1.0, "entity_key": "e", "cluster_key": "k", "label": 0})
    f = build_features(df, WINDOWS, 50)
    assert list(f["hist_n"]) == [0, 0, 2]                      # ties are not "earlier"
    assert list(f["cnt_3600"]) == [0, 0, 2]
    assert f["cluster_cnt_3600"].iat[1] == 0


def test_no_label_or_outcome_in_model_inputs(tiny_feat):
    names = fit_space(tiny_feat.iloc[:2000]).names
    assert not any("label" in n or "fraud" in n for n in names)


def test_folds_are_chronological_and_validated(tiny_df):
    for fold in chronological_folds(len(tiny_df), [[0.5, 0.65, 0.8], [0.6, 0.75, 0.9], [0.7, 0.85, 1.0]]):
        assert_chronological(tiny_df, fold)
    bad = {"train": np.arange(0, 100), "val": np.arange(90, 150), "test": np.arange(150, 200)}
    with pytest.raises(AssertionError):
        assert_chronological(tiny_df, bad)
    shuffled = tiny_df.sample(frac=1, random_state=0).reset_index(drop=True)
    with pytest.raises(AssertionError):                            # unsorted time must fail loudly
        assert_chronological(shuffled, chronological_folds(len(shuffled), [[0.5, 0.65, 0.8]])[0])


def test_encoder_is_fitted_on_train_only():
    tr = pd.DataFrame({"c": ["a", "a", "b"]})
    enc = TargetEncoder(["c"], smoothing=1.0).fit(tr, np.array([1, 1, 0]))
    out = enc.transform(pd.DataFrame({"c": ["a", "zzz"]}))
    assert out["c_te"].iat[1] == pytest.approx(enc.prior)          # unseen category -> train prior, never test info
    assert set(enc.maps["c"]) == {"a", "b"}


def test_shuffled_time_collapses_performance(tiny_df):
    """If behavioural history is real signal, destroying time order must hurt. Fails loudly otherwise."""
    cfg_xgb = {"n_estimators": 80, "max_depth": 4, "learning_rate": 0.15, "early_stopping_rounds": 10}

    def score(df):
        feat = build_features(df, WINDOWS, 50)
        n = len(feat)
        tr, va, te = feat.iloc[: int(n * .6)], feat.iloc[int(n * .6): int(n * .75)], feat.iloc[int(n * .75):]
        sp = fit_space(tr)
        m = train_xgb(sp.matrix(tr), tr.label.to_numpy(), sp.matrix(va), va.label.to_numpy(), cfg_xgb, True, 0)
        return average_precision_score(te.label, m.raw(sp.matrix(te)))

    real = score(tiny_df)
    shuf = tiny_df.copy()
    shuf["ts"] = np.random.default_rng(1).permutation(tiny_df["ts"].to_numpy())          # break time <-> behaviour link
    shuf = shuf.sort_values("ts").reset_index(drop=True)
    assert real > score(shuf) + 0.03


def test_determinism_same_seed_same_features(tiny_df, tiny_feat):
    again = build_features(tiny_df, WINDOWS, 50)
    pd.testing.assert_frame_equal(tiny_feat, again)
