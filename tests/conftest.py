"""Shared fixtures: a tiny synthetic dataset and a tiny end-to-end experiment loaded into a temp database."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml import synth  # noqa: E402
from ml.config import load_config  # noqa: E402
from ml.features import build_features  # noqa: E402

TINY = {
    "name": "pytest",
    "include": str(ROOT / "configs" / "default.yaml"),
    "seeds": [3],
    "overrides": {
        "data": {"synthetic": {"n_entities": 140, "n_days": 24, "n_rings": 9}},
        "models": {"xgb": {"n_estimators": 60, "max_depth": 4, "learning_rate": 0.15, "subsample": 0.8,
                           "colsample_bytree": 0.8, "min_child_weight": 1, "early_stopping_rounds": 10}},
        "splits": {"folds": [[0.5, 0.7, 1.0]]},
        "stability": {"M": 6},
        "ranking": {"grid": {"beta": [0.0, 0.5], "gamma": [0.0, 0.5], "lam": [0.0, 0.4]}},
        "evaluation": {"bootstrap_n": 60, "deletion_n": 30},
    },
}


@pytest.fixture(scope="session")
def tiny_df():
    return synth.generate(140, 24, 9, seed=3)


@pytest.fixture(scope="session")
def tiny_feat(tiny_df):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return build_features(tiny_df, cfg["features"]["windows_s"], cfg["features"]["history_cap"])


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    p = tmp_path_factory.mktemp("cfg") / "tiny.yaml"
    p.write_text(yaml.safe_dump(TINY), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def loaded_db(tiny_cfg, tmp_path_factory):
    """Runs the real pipeline end to end on tiny data and returns the sqlite URL."""
    from ml.run_experiment import main
    url = f"sqlite:///{(tmp_path_factory.mktemp('db') / 'eg.db').as_posix()}"
    run_id = "pytest-run"
    main(["--config", str(tiny_cfg), "--db", url, "--run-id", run_id])
    yield url
    shutil.rmtree(ROOT / "results" / run_id, ignore_errors=True)


@pytest.fixture()
def client(loaded_db):
    from fastapi.testclient import TestClient

    from api import core, db
    from api.main import app
    db.init(loaded_db)
    core.clear_cache()
    with TestClient(app) as c:
        yield c
