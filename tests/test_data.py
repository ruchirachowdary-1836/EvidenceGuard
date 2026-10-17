"""Dataset loaders (tiny fixtures: the real Kaggle / PaySim files are not in the repo) and database copy."""
import pandas as pd
import pytest
from sqlalchemy import func, select

from ml import copy_db, store
from ml.data import CANONICAL, check_canonical, from_ieee_cis, from_paysim


def test_ieee_cis_loader_maps_to_canonical_schema(tmp_path):
    pd.DataFrame({
        "TransactionID": [3, 1, 2], "TransactionDT": [300, 100, 200], "TransactionAmt": [10.0, 20.0, 30.0], "ProductCD": ["W", "C", "W"],
        "card6": ["credit", None, "debit"], "P_emaildomain": ["gmail.com", None, "yahoo.com"], "D1": [1.0, None, 3.0],
        "card1": [111, 111, 222], "addr1": [10.0, 10.0, None], "isFraud": [0, 1, 0], **{f"V{i}": [0.1, None, 0.3] for i in range(1, 7)},
    }).to_csv(tmp_path / "train_transaction.csv", index=False)
    pd.DataFrame({"TransactionID": [1], "DeviceType": ["mobile"], "DeviceInfo": ["SM-G960"]}).to_csv(tmp_path / "train_identity.csv", index=False)
    df = from_ieee_cis(tmp_path).sort_values("ts").reset_index(drop=True)
    check_canonical(df)
    assert set(CANONICAL) <= set(df.columns) and df["ts"].tolist() == [100, 200, 300]
    assert df.loc[0, "entity_key"] == "111_10" and df.loc[0, "device_type"] == "mobile" and df.loc[1, "entity_key"] == "222_-1"
    assert df["label"].tolist() == [1, 0, 0]


def test_paysim_loader_uses_accounts_as_entity_and_cluster(tmp_path):
    pd.DataFrame({"step": [2, 1], "type": ["TRANSFER", "CASH_OUT"], "amount": [5.0, 6.0], "nameOrig": ["C1", "C2"], "nameDest": ["C9", "C9"],
                  "oldbalanceOrg": [10.0, 20.0], "newbalanceOrig": [5.0, 14.0], "isFraud": [1, 0]}).to_csv(tmp_path / "PS_x.csv", index=False)
    df = from_paysim(tmp_path)
    assert df["ts"].tolist() == [7200, 3600] and set(df["cluster_key"]) == {"C9"} and set(df["product"]) == {"TRANSFER", "CASH_OUT"}


def test_check_canonical_rejects_bad_frames():
    df = pd.DataFrame({"txn_id": [1, 1], "ts": [1, 2]})
    with pytest.raises(ValueError, match="missing columns"):
        check_canonical(df)


def test_copy_db_round_trips_every_table(loaded_db, tmp_path):
    dst = f"sqlite:///{(tmp_path / 'copy.db').as_posix()}"
    counts = copy_db.copy(loaded_db, dst)
    assert counts["scores"] > 0 and counts["explanations"] > 0
    src, out = store.make_engine(loaded_db), store.make_engine(dst)
    for t in (store.transactions, store.scores, store.alerts, store.stability_runs, store.experiment_runs):
        with src.connect() as a, out.connect() as b:
            assert a.execute(select(func.count()).select_from(t)).scalar_one() == b.execute(select(func.count()).select_from(t)).scalar_one()
    assert copy_db.copy(loaded_db, dst, if_empty=True) == {"skipped": 1}
