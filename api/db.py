"""Database access for the API. Reads may use a read-only role (DATABASE_URL_READONLY)."""
from __future__ import annotations

import os

from sqlalchemy.engine import Engine

from ml.store import DEFAULT_URL, make_engine

_write: Engine | None = None
_read: Engine | None = None


def init(url: str | None = None, read_url: str | None = None) -> None:
    global _write, _read
    url = url or os.getenv("DATABASE_URL", DEFAULT_URL)
    read_url = read_url or os.getenv("DATABASE_URL_READONLY") or url
    _write = make_engine(url)
    _read = _write if read_url == url else make_engine(read_url)


def writer() -> Engine:
    if _write is None:
        init()
    return _write  # type: ignore[return-value]


def reader() -> Engine:
    if _read is None:
        init()
    return _read  # type: ignore[return-value]
