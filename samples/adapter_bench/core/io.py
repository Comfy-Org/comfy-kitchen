"""Rows on disk: {"meta": {...}, "rows": [...]} per family per device."""

import json
import math
from pathlib import Path


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def save_rows(path: Path, meta: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"meta": meta, "rows": [{k: _clean(v) for k, v in r.items()} for r in rows]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["meta"], data["rows"]
