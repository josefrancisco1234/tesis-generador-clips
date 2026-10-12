"""Utilidades de E/S: JSON atómico y hashing estable."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_default)
    os.replace(tmp, path)


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return h.hexdigest()[:24]
