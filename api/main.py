"""EvidenceGuard API: ranked budgeted alert queue, evidence, simulator, experiments, audit trail."""
from __future__ import annotations

import os
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from api import core, db
from api.routers import alerts, audit, experiments, simulate, stream
from ml.store import md

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

@asynccontextmanager
async def lifespan(_: FastAPI):
    md.create_all(db.writer())
    yield


app = FastAPI(title="EvidenceGuard", version="1.0.0", lifespan=lifespan,
              description="Research prototype. Data is synthetic or public benchmark data - never real UPI traffic.")
origins = [o.strip() for o in os.getenv("WEB_ORIGIN", "http://localhost:5173,http://localhost:8080,http://localhost:8000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Analyst"])
for r in (alerts.router, simulate.router, experiments.router, audit.router, stream.router):
    app.include_router(r)


@app.get("/api/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/api/meta", tags=["meta"])
def meta():
    pool = core.get_pool()
    m = pool.metrics
    return {"run_id": pool.run_id, "model_version": m.get("model_version"), "dataset": m.get("dataset"), "days": pool.days, "default_day": pool.default_day,
            "methods": list(core.METHODS), "modes": list(core.MODES), "defaults": pool.defaults, "pool_size": int(len(pool.df)),
            "n_day": pool.n_day, "default_budget": 0.05,
            "caveat": "Research prototype on " + ("synthetic data" if (m.get("dataset") or {}).get("source") == "synthetic" else "public benchmark data")
                      + ". Not real Indian UPI traffic; no claim of UPI effectiveness."}


if WEB_DIST.exists():                       # single-process mode: the API also serves the built dashboard
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = WEB_DIST / path
        return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")
