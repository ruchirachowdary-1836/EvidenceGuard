from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from api import db
from ml.store import audit_log

router = APIRouter(prefix="/api", tags=["audit"])


def _row(r) -> dict:
    return {"id": r["id"], "at": str(r["at"]), "actor": r["actor"], "action": r["action"], "detail": r["detail"]}


@router.get("/audit")
def audit(page: int = Query(1, ge=1), size: int = Query(25, ge=1, le=200), action: str | None = None, actor: str | None = None):
    q = select(audit_log)
    cnt = select(func.count()).select_from(audit_log)
    for col, val in ((audit_log.c.action, action), (audit_log.c.actor, actor)):
        if val:
            q, cnt = q.where(col == val), cnt.where(col == val)
    with db.reader().connect() as conn:
        total = conn.execute(cnt).scalar_one()
        rows = conn.execute(q.order_by(audit_log.c.id.desc()).limit(size).offset((page - 1) * size)).mappings().all()
    return {"page": page, "size": size, "total": total, "items": [_row(r) for r in rows]}


def _safe(v) -> str:
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in "=+-@" else s            # neutralise spreadsheet formula injection


@router.get("/audit/export.csv")
def export_csv():
    with db.reader().connect() as conn:
        rows = conn.execute(select(audit_log).order_by(audit_log.c.id)).mappings().all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "at", "actor", "action", "detail"])
    for r in rows:
        w.writerow([r["id"], r["at"], _safe(r["actor"]), _safe(r["action"]), _safe(json.dumps(r["detail"]))])
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="audit_trail.csv"'})
