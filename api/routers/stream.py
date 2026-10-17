"""Server-sent events: a paced replay of held-out candidate alerts so the dashboard can show a live feed."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from api import core

router = APIRouter(prefix="/api", tags=["live"])


async def replay(interval: float, limit: int | None) -> AsyncIterator[str]:
    pool = core.get_pool()
    df = pool.df.sort_values("ts")
    n = 0
    yield f"event: hello\ndata: {json.dumps({'run_id': pool.run_id, 'total': len(df), 'note': 'replay of held-out test transactions, not live traffic'})}\n\n"
    while True:
        for row in df.itertuples():
            ev = {"alert_id": int(row.alert_id), "txn_id": int(row.txn_id), "p": float(row.p), "a": float(row.a), "e": float(row.e),
                  "amount": float(row.amount), "ts": int(row.ts), "day": row.day, "cluster_key": row.cluster_key}
            yield f"event: txn\ndata: {json.dumps(ev)}\n\n"
            n += 1
            if limit and n >= limit:
                return
            await asyncio.sleep(interval)


@router.get("/stream")
async def stream(interval: float = Query(1.0, ge=0.02, le=10), limit: int | None = Query(None, ge=1, le=100000)):
    return StreamingResponse(replay(interval, limit), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
