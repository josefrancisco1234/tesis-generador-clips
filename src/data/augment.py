"""Añade a las tablas featurizadas: grupos E (eventos de audio) y L (lingüísticos), y la etiqueta
de clips humanos de Repurpose-10K cuando existe (y_human_overlap, y_human). D-023, D-024.

    python -m src.data.augment [--no-events]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.common.config import Config, load_config
from src.common.io import read_json
from src.common.logging_utils import get_logger, setup_logging
from src.data.featurize import KINDS, features_path
from src.data.repurpose import clips_path
from src.data.youtube_dataset import transcript_path, wav_path
from src.features.extra import audio_events, augment_table
from src.transcribe.whisper import Transcript

log = get_logger(__name__)


def human_overlap(df: pd.DataFrame, segments: list[list[float]]) -> np.ndarray:
    segs = np.array(segments) if segments else np.zeros((0, 2))
    out = np.zeros(len(df))
    for i, (s, e) in enumerate(zip(df["start"], df["end"])):
        inter = np.clip(np.minimum(segs[:, 1], e) - np.maximum(segs[:, 0], s), 0, None).sum() if len(segs) else 0.0
        out[i] = min(1.0, inter / max(e - s, 1e-3))
    return out


def augment_video(cfg: Config, vid: str, events: bool, stride: float) -> bool:
    paths = {k: features_path(cfg, k, vid) for k in KINDS}
    if not all(p.exists() for p in paths.values()):
        return False
    tr = Transcript.load(transcript_path(cfg, vid, cfg.transcribe["model_dataset"]))
    ev = None
    wav = wav_path(cfg, vid)
    if events and wav.exists():
        ev = audio_events(wav, cfg.path("interim") / "audio_events" / f"{vid}.parquet",
                          str(cfg.path("models_ext") / "panns" / "Cnn14_mAP=0.431.pth"), stride)
    hc = clips_path(cfg, vid)
    segs = read_json(hc)["segments"] if hc.exists() else None
    for k, p in paths.items():
        df = augment_table(pd.read_parquet(p), tr, ev)
        if segs is not None:
            df["y_human_overlap"] = human_overlap(df, segs)
            df["y_human"] = (df["y_human_overlap"] >= 0.5).astype(int)
        df.to_parquet(p, index=False)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--stride", type=float, default=3.0)
    ap.add_argument("--only-missing", action="store_true", help="salta videos que ya tienen columnas e_")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(cfg.logging["level"], cfg.logging["file"])
    vids = sorted(p.stem for p in (cfg.path("processed") / "features" / "sentence").glob("*.parquet"))
    for i, v in enumerate(vids, 1):
        if args.only_missing:
            cols = pd.read_parquet(features_path(cfg, "sentence", v), columns=None).columns
            if any(c.startswith("e_") for c in cols):
                continue
        t = time.time()
        ok = augment_video(cfg, v, not args.no_events, args.stride)
        log.info("augment [%d/%d] %s ok=%s %.0fs", i, len(vids), v, ok, time.time() - t)


if __name__ == "__main__":
    main()
