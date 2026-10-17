"""YAML config loading with a tiny `include` + `overrides` mechanism."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "include" in cfg:
        base = load_config(path.parent / cfg.pop("include"))
        overrides = cfg.pop("overrides", {})
        cfg = _merge(_merge(base, overrides), cfg)
    return cfg
