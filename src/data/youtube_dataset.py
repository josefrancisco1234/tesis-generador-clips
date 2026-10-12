"""Construcción del dataset propio (YT-MR-ES/EN): videos hablados con curva Most Replayed.

Pasos reanudables:
  1. manifest: búsqueda → filtro por duración → metadatos → sólo videos con heatmap.
  2. acquire: audio → wav 16 kHz → transcripción faster-whisper → verificación de idioma.
Sólo se versiona el manifest (IDs + metadatos); audio y transcripciones quedan en data_root.
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd

from src.common.config import Config, load_config, set_seeds
from src.common.io import read_json, write_json
from src.common.logging_utils import get_logger, setup_logging
from src.ingest import media, youtube
from src.transcribe.whisper import transcribe

log = get_logger(__name__)
MAX_PER_CHANNEL = 2


def heatmap_path(cfg: Config, vid: str) -> Path:
    return cfg.path("interim") / "heatmaps" / f"{vid}.json"


def transcript_path(cfg: Config, vid: str, model: str) -> Path:
    return cfg.path("interim") / "transcripts" / model / f"{vid}.json"


def wav_path(cfg: Config, vid: str) -> Path:
    return cfg.path("raw") / "wav" / f"{vid}.wav"


def build_manifest(cfg: Config, out: Path) -> pd.DataFrame:
    dcfg = cfg.dataset_youtube
    rows: list[dict] = []
    if out.exists():
        rows = pd.read_csv(out).to_dict("records")
    seen = {r["video_id"] for r in rows}
    per_channel: dict[str, int] = {}
    for r in rows:
        per_channel[r["channel_id"]] = per_channel.get(r["channel_id"], 0) + 1

    for lang, queries in dcfg["queries"].items():
        target = dcfg["target_videos"][lang]
        have = sum(1 for r in rows if r["query_lang"] == lang)
        for q in queries:
            if have >= target:
                break
            for hit in youtube.search(q, dcfg["per_query"]):
                if have >= target:
                    break
                vid, dur = hit["video_id"], hit.get("duration") or 0
                if vid in seen or not (dcfg["min_duration_s"] <= dur <= dcfg["max_duration_s"]):
                    continue
                if per_channel.get(hit.get("channel_id"), 0) >= MAX_PER_CHANNEL:
                    continue
                seen.add(vid)
                youtube.polite_sleep(cfg.ingest["sleep_between_requests"])
                meta = youtube.fetch_metadata(vid)
                if not meta or not meta.get("heatmap"):
                    log.info("sin heatmap: %s", vid)
                    continue
                write_json(heatmap_path(cfg, vid), meta)
                per_channel[meta["channel_id"]] = per_channel.get(meta["channel_id"], 0) + 1
                rows.append({"video_id": vid, "query_lang": lang, "query": q, "title": meta["title"],
                             "duration": meta["duration"], "channel_id": meta["channel_id"],
                             "channel": meta["channel"], "view_count": meta["view_count"],
                             "upload_date": meta["upload_date"], "n_bins": len(meta["heatmap"])})
                have += 1
                log.info("[%s %d/%d] %s — %s", lang, have, target, vid, meta["title"])
                pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    log.info("manifest con %d videos → %s", len(df), out)
    return df


def _try_lock(path: Path) -> bool:
    """Lock exclusivo por video para poder correr varios workers en paralelo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def acquire(cfg: Config, manifest: pd.DataFrame, model: str, reverse: bool = False) -> pd.DataFrame:
    tcfg = cfg.transcribe
    status = []
    rows = list(manifest.itertuples())
    seguidos = 0                                               # bloqueos anti-bot consecutivos (D-042)
    max_seguidos = int(cfg.ingest.get("block_max_consecutive", 2))
    pausa_s = float(cfg.ingest.get("block_pause_s", 1800))
    for i, row in enumerate(rows[::-1] if reverse else rows, 1):
        vid = row.video_id
        youtube.ultimo_error = ""            # no arrastrar el error de otro video
        tp = transcript_path(cfg, vid, model)
        ok, lang = False, None
        lock = tp.with_suffix(".lock")
        if not tp.exists() and not _try_lock(lock):
            log.info("%s lo está procesando otro worker; se omite", vid)
            status.append({"video_id": vid, "ok": None, "detected_lang": None})
            continue
        try:
            if not heatmap_path(cfg, vid).exists():
                meta = youtube.fetch_metadata(vid)
                if meta and meta.get("heatmap"):
                    write_json(heatmap_path(cfg, vid), meta)
            wav = wav_path(cfg, vid)
            if not wav.exists() and not tp.exists():
                base = cfg.ingest["sleep_between_requests"]
                youtube.polite_sleep(base * 2, base)  # 4–6 s con la config por defecto; reduce bloqueos anti-bot
                youtube.ultimo_error = ""
                src = youtube.download_audio(vid, cfg.path("raw") / "audio", cfg.ingest["youtube_audio_format"])
                if src is None:
                    raise RuntimeError("audio no disponible")
                media.extract_audio(src, wav, cfg.ingest["audio_sr"])
                src.unlink(missing_ok=True)  # se conserva sólo el wav derivado
            tr = transcribe(wav, tp, model, tcfg["compute_type"], tcfg["cpu_threads"], tcfg["beam_size"],
                            tcfg["vad_filter"], tcfg["language"], tcfg.get("device", "auto"))
            lang, ok = tr.language, tr.language == row.query_lang and len(tr.words) > 500
        except Exception as exc:
            log.warning("falló %s: %s", vid, exc)
        finally:
            lock.unlink(missing_ok=True)
        if not ok and youtube.es_bloqueo(youtube.ultimo_error):
            seguidos += 1
            if seguidos >= max_seguidos:
                log.warning("%d bloqueos anti-bot seguidos: pausa de %.0f min (los videos fallidos se reintentan "
                            "al volver a ejecutar la adquisición)", seguidos, pausa_s / 60)
                youtube.polite_sleep(pausa_s)
                seguidos = 0
        else:
            seguidos = 0                     # cualquier resultado que no sea bloqueo reinicia la cuenta
        status.append({"video_id": vid, "ok": ok, "detected_lang": lang})
        log.info("[%d/%d] %s ok=%s idioma=%s", i, len(manifest), vid, ok, lang)
    st = pd.DataFrame(status)
    return manifest.merge(st, on="video_id", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--step", choices=["manifest", "acquire", "all"], default="all")
    ap.add_argument("--reverse", action="store_true", help="recorre el manifest al revés (segundo worker)")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, {"transcribe": {"cpu_threads": args.threads}} if args.threads else None)
    setup_logging(cfg.logging["level"], cfg.logging["file"])
    set_seeds(cfg.seed)
    out = Path(args.manifest) if args.manifest else cfg.path("manifests") / "youtube_mr.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.step in ("manifest", "all"):
        df = build_manifest(cfg, out)
    else:
        df = pd.read_csv(out)
    if args.step in ("acquire", "all"):
        df = acquire(cfg, df, cfg.transcribe["model_dataset"], args.reverse)
        if not args.reverse:
            df.to_csv(out.with_name(out.stem + "_status.csv"), index=False)
        log.info("adquisición terminada: %d/%d válidos (%s)", int(df["ok"].fillna(False).astype(bool).sum()), len(df), date.today())


if __name__ == "__main__":
    main()
