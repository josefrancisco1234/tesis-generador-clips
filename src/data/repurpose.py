"""Repurpose-10K (Wu et al., AAAI 2025; github.com/yongliang-wu/Repurpose, licencia MIT).

Videos largos de YouTube con clips elegidos por personas (curación en la plataforma de Opus Clip +
refinamiento manual). Se usan como SEGUNDA etiqueta independiente: para los videos que además
tienen curva Most Replayed, se puede medir (a) el acuerdo entre la re-reproducción del público
y los clips humanos, y (b) si el ranker recupera los clips humanos. D-023.

    python -m src.data.repurpose --target 80
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.io import read_json, write_json
from src.common.logging_utils import get_logger, setup_logging
from src.data.youtube_dataset import MAX_PER_CHANNEL, heatmap_path
from src.ingest import youtube

log = get_logger(__name__)
SRC = Path("data/external/repurpose")


def load_clips(root: Path = SRC) -> dict[str, dict]:
    """youtube_id → {'segments': [[s, e], ...] absolutos, 'covered_until': s}."""
    out: dict[str, dict] = defaultdict(lambda: {"segments": [], "covered_until": 0.0})
    for split in ("train", "val", "test"):
        for e in json.load(open(root / f"{split}.json", encoding="utf-8")):
            v = out[e["youtube_id"]]
            v["segments"] += [list(map(float, s)) for s in e["segments"]]
            v["covered_until"] = max(v["covered_until"], float(e["timeRange"][1]))
            v.setdefault("splits", set()).add(split)
    return out


def clips_path(cfg: Config, vid: str) -> Path:
    return cfg.path("interim") / "human_clips" / f"{vid}.json"


def build(cfg: Config, manifest: Path, target: int, min_min: float = 20, max_min: float = 75) -> pd.DataFrame:
    clips = load_clips()
    rows = pd.read_csv(manifest).to_dict("records") if manifest.exists() else []
    seen = {r["video_id"] for r in rows}
    per_channel: dict[str, int] = defaultdict(int)
    for r in rows:
        per_channel[r["channel_id"]] += 1
    have = sum(1 for r in rows if r.get("query") == "repurpose10k")
    pool = sorted(v for v, c in clips.items() if min_min * 60 <= c["covered_until"] <= max_min * 60 and v not in seen)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(pool)
    tried = 0
    for vid in pool:
        if have >= target:
            break
        tried += 1
        youtube.polite_sleep(cfg.ingest["sleep_between_requests"] * 1.5)
        meta = youtube.fetch_metadata(vid)
        if not meta or not meta.get("heatmap"):
            continue
        if per_channel[meta["channel_id"]] >= MAX_PER_CHANNEL or not (min_min * 60 <= (meta["duration"] or 0) <= max_min * 60):
            continue
        lang = (meta.get("language") or "").lower()
        if lang and not lang.startswith("en"):
            continue
        write_json(heatmap_path(cfg, vid), meta)
        c = clips[vid]
        write_json(clips_path(cfg, vid), {"segments": c["segments"], "splits": sorted(c["splits"])})
        per_channel[meta["channel_id"]] += 1
        rows.append({"video_id": vid, "query_lang": "en", "query": "repurpose10k", "title": meta["title"],
                     "duration": meta["duration"], "channel_id": meta["channel_id"], "channel": meta["channel"],
                     "view_count": meta["view_count"], "upload_date": meta["upload_date"], "n_bins": len(meta["heatmap"])})
        have += 1
        log.info("[repurpose %d/%d] %s — %s (%d clips humanos)", have, target, vid, meta["title"], len(c["segments"]))
        pd.DataFrame(rows).to_csv(manifest, index=False)
    log.info("repurpose: %d aceptados de %d consultados", have, tried)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=80)
    ap.add_argument("--manifest", default="data/manifests/youtube_mr_repurpose.csv")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(cfg.logging["level"], cfg.logging["file"])
    build(cfg, Path(args.manifest), args.target)


if __name__ == "__main__":
    main()
