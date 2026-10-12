"""Features + etiquetas por video para el dataset propio (reanudable, un parquet por video)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.common.config import Config, load_config, set_seeds
from src.common.io import read_json
from src.common.logging_utils import get_logger, setup_logging
from src.data.youtube_dataset import heatmap_path, transcript_path, wav_path
from src.features.audio import load_or_compute
from src.features.builder import build_context, windows_frame
from src.ranking.labels import add_labels
from src.segment.candidates import attach_sentences, bin_windows, fixed_windows, sentence_candidates
from src.segment.sentences import split_sentences
from src.transcribe.whisper import Transcript

log = get_logger(__name__)
KINDS = ("sentence", "bin", "fixed")


def features_path(cfg: Config, kind: str, vid: str) -> Path:
    return cfg.path("processed") / "features" / kind / f"{vid}.parquet"


def featurize_video(cfg: Config, row: dict, model: str, force: bool = False) -> bool:
    vid = row["video_id"]
    if not force and all(features_path(cfg, k, vid).exists() for k in KINDS):
        return True
    tp, hp = transcript_path(cfg, vid, model), heatmap_path(cfg, vid)
    if not tp.exists() or not hp.exists():
        log.warning("faltan insumos para %s", vid)
        return False
    t0 = time.time()
    tr = Transcript.load(tp)
    heat = read_json(hp)["heatmap"]
    sc = cfg.segment
    sents = split_sentences(tr, sc["sentence_max_words"], sc["sentence_pause_split_s"])
    if len(sents) < 20:
        log.warning("muy pocas oraciones en %s (%d)", vid, len(sents))
        return False
    wav = wav_path(cfg, vid)
    audio = (load_or_compute(wav, cfg.path("interim") / "audio_frames" / f"{vid}.npz", cfg.ingest["audio_sr"],
                             cfg.features["audio_hop"]) if wav.exists() else None)
    ctx = build_context(tr, sents, cfg, audio, cfg.path("cache"), f"{vid}_{model}")
    windows = {
        "sentence": sentence_candidates(sents, sc["min_clip_s"], sc["max_clip_s"], sc["target_durations"],
                                        sc["train_start_stride"], sc["max_candidates_per_video_train"], cfg.seed),
        "bin": attach_sentences(bin_windows(heat), sents),
        "fixed": attach_sentences(fixed_windows(tr.duration, sc["fixed_window_s"]), sents),
    }
    meta = {"video_id": vid, "channel_id": row.get("channel_id"), "lang": tr.language,
            "query_lang": row.get("query_lang"), "duration_video": tr.duration}
    lc = cfg.labels
    for kind, wins in windows.items():
        df = windows_frame(ctx, wins, cfg.features["hook_window_s"], cfg.features["hook_words"], meta)
        df = add_labels(df, wins, heat, lc["max_n_gt"], lc["gt_threshold"], lc["n_grades"], lc["relevant_overlap"])
        out = features_path(cfg, kind, vid)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    log.info("features %s: %d cand, %d bins, %d fijas en %.0fs", vid, len(windows["sentence"]),
             len(windows["bin"]), len(windows["fixed"]), time.time() - t0)
    return True


def load_table(cfg: Config, kind: str, video_ids: list[str] | None = None) -> pd.DataFrame:
    files = sorted((cfg.path("processed") / "features" / kind).glob("*.parquet"))
    if video_ids is not None:
        keep = set(video_ids)
        files = [f for f in files if f.stem in keep]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--status", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging["level"], cfg.logging["file"])
    set_seeds(cfg.seed)
    status = Path(args.status) if args.status else cfg.path("manifests") / "youtube_mr_status.csv"
    df = pd.read_csv(status)
    df = df[df["ok"] == True]  # noqa: E712
    n_ok = sum(featurize_video(cfg, r, cfg.transcribe["model_dataset"], args.force) for r in df.to_dict("records"))
    log.info("featurización: %d/%d videos", n_ok, len(df))


if __name__ == "__main__":
    main()
