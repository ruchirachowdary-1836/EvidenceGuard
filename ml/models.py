"""Baseline fraud score p_i: logistic regression + XGBoost, with validation-set calibration.

Everything that learns from data (target encoding, scaling, imputation, calibration) is fitted on the
training window (calibration on validation, as the plan specifies) and then only applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import CAT_COLS, numeric_columns


class TargetEncoder:
    """Smoothed target mean per category, fitted on the training window only."""

    def __init__(self, cols: list[str], smoothing: float = 20.0):
        self.cols, self.m = cols, smoothing
        self.maps: dict[str, dict] = {}
        self.prior = 0.0

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "TargetEncoder":
        self.prior = float(np.mean(y))
        for c in self.cols:
            g = pd.DataFrame({"c": df[c].to_numpy(), "y": y}).groupby("c")["y"].agg(["sum", "count"])
            self.maps[c] = ((g["sum"] + self.m * self.prior) / (g["count"] + self.m)).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"{c}_te": df[c].map(self.maps[c]).fillna(self.prior).to_numpy() for c in self.cols},
                            index=df.index)


@dataclass
class FeatureSpace:
    names: list[str]
    encoder: TargetEncoder
    medians: pd.Series = field(default_factory=pd.Series)

    def matrix(self, feat: pd.DataFrame) -> np.ndarray:
        X = pd.concat([feat[[c for c in self.names if c in feat.columns]],
                       self.encoder.transform(feat)], axis=1)[self.names]
        return X.fillna(self.medians).to_numpy(np.float32)


def fit_space(train_feat: pd.DataFrame) -> FeatureSpace:
    y = train_feat["label"].to_numpy()
    enc = TargetEncoder(CAT_COLS).fit(train_feat, y)
    names = numeric_columns(train_feat) + [f"{c}_te" for c in CAT_COLS]
    num = pd.concat([train_feat[numeric_columns(train_feat)], enc.transform(train_feat)], axis=1)
    return FeatureSpace(names=names, encoder=enc, medians=num.median())


@dataclass
class Fitted:
    kind: str
    model: object
    scaler: StandardScaler | None = None

    def raw(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "logreg":
            return self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return self.model.predict_proba(X)[:, 1]

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Unbounded log-odds: keeps ordering where float probabilities saturate at 0 / 1."""
        if self.kind == "logreg":
            return self.model.decision_function(self.scaler.transform(X))
        return self.model.get_booster().predict(xgb.DMatrix(X), output_margin=True)


def train_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 0) -> Fitted:
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=seed)
    lr.fit(sc.transform(X), y)
    return Fitted("logreg", lr, sc)


def train_xgb(X, y, Xv, yv, params: dict, weighted: bool = True, seed: int = 0) -> Fitted:
    params = dict(params)
    esr = params.pop("early_stopping_rounds", 30)
    spw = float((y == 0).sum() / max((y == 1).sum(), 1)) if weighted else 1.0
    m = xgb.XGBClassifier(**params, scale_pos_weight=spw, eval_metric="aucpr", early_stopping_rounds=esr,
                          tree_method="hist", n_jobs=4, random_state=seed, verbosity=0)
    m.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    return Fitted("xgb_w" if weighted else "xgb", m)


class Calibrator:
    """Probability calibration fitted on VALIDATION margins (the ranker mixes p with other scores).

    platt (default): logistic regression on the log-odds margin - smooth and strictly monotone, so it never
    creates ties. isotonic: step function, available for comparison but saturates at 0 / 1 on small windows.
    """

    def __init__(self, method: str = "platt"):
        self.method = method
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0) if method == "isotonic" else LogisticRegression(C=1e4, max_iter=1000)

    def fit(self, margin: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "isotonic":
            self.model.fit(margin, y)
        else:
            self.model.fit(np.asarray(margin).reshape(-1, 1), y)
        return self

    def __call__(self, margin: np.ndarray) -> np.ndarray:
        m = np.asarray(margin, dtype=float)
        return self.model.predict(m) if self.method == "isotonic" else self.model.predict_proba(m.reshape(-1, 1))[:, 1]
