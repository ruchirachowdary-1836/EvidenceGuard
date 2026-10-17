"""Copy every table from one database to another (e.g. the shipped SQLite demo -> PostgreSQL).

    python -m ml.copy_db --src sqlite:///data/demo.db --dst postgresql+psycopg://user:pw@host/db [--if-empty]
"""
from __future__ import annotations

import argparse

from sqlalchemy import func, insert, select, text

from . import store

ORDER = [store.transactions, store.experiment_runs, store.scores, store.explanations, store.stability_runs, store.alerts,
         store.decisions, store.audit_log]


def copy(src_url: str, dst_url: str, if_empty: bool = False, chunk: int = 5000) -> dict[str, int]:
    src, dst = store.make_engine(src_url), store.make_engine(dst_url)
    store.md.create_all(dst)
    counts: dict[str, int] = {}
    with dst.begin() as d, src.connect() as s:
        if if_empty and d.execute(select(func.count()).select_from(store.experiment_runs)).scalar_one() > 0:
            return {"skipped": 1}
        for t in ORDER:
            rows = [dict(r) for r in s.execute(select(t)).mappings()]
            for i in range(0, len(rows), chunk):
                d.execute(insert(t), rows[i:i + chunk])
            counts[t.name] = len(rows)
        if dst.dialect.name == "postgresql":              # keep serial sequences ahead of the copied ids
            for t, col in ((store.alerts, "alert_id"), (store.decisions, "id"), (store.audit_log, "id"), (store.stability_runs, "id")):
                d.execute(text(f"SELECT setval(pg_get_serial_sequence('{t.name}', '{col}'), COALESCE((SELECT MAX({col}) FROM {t.name}), 1))"))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--if-empty", action="store_true", help="do nothing when the destination already has a run")
    a = ap.parse_args()
    print(copy(a.src, a.dst, a.if_empty))


if __name__ == "__main__":
    main()
