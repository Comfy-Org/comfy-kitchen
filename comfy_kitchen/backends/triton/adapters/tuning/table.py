# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The on-disk tuning cache: rows of (kernel, cat, num) -> config, one file per arch."""

import json
import os
from pathlib import Path

VERSION = 2


def read_rows(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if data.get("version") != VERSION:
        return []
    return data.get("entries", [])


def write_rows(path: Path, rows: list[dict], meta: dict) -> None:
    """Whole-file write through a temp file and an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps({"version": VERSION, **meta, "entries": rows}, indent=1),
        encoding="utf-8",
    )
    os.replace(tmp, path)
