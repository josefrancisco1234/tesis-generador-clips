"""Acceso a YouTube vía yt-dlp: búsqueda, metadatos (curva Most Replayed) y descarga.

Uso académico: sólo se guardan IDs y datos derivados; el audio se puede borrar tras
extraer features (ver docs/DECISIONES.md, D-009).
"""
from __future__ import annotations

import random
import re
import time
from pathlib import Path
from typing import Any

from src.common.logging_utils import get_logger

log = get_logger(__name__)

# yt-dlp traga las excepciones y devuelve None; se guarda el último motivo para poder distinguir un
# bloqueo anti-bot (hay que pausar) de un video no disponible (se salta). D-042.
ultimo_error: str = ""
_BLOQUEO = re.compile(r"confirm you.{0,4}re not a bot|sign in to confirm|http error 429|too many requests|"
                      r"rate.?limit", re.IGNORECASE)


def es_bloqueo(msg: str | None) -> bool:
    """¿El mensaje de error de yt-dlp corresponde a un bloqueo anti-bot o a un límite de peticiones?"""
    return bool(msg and _BLOQUEO.search(msg))


def _ydl(opts: dict[str, Any]):
    import yt_dlp

    base = {"quiet": True, "no_warnings": True, "noprogress": True, "retries": 3}
    base.update(opts)
    return yt_dlp.YoutubeDL(base)


def search(query: str, n: int) -> list[dict]:
    """Búsqueda plana (no descarga nada). Devuelve id, título, duración, canal."""
    with _ydl({"extract_flat": True}) as ydl:
        res = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    out = []
    for e in res.get("entries") or []:
        if not e or not e.get("id"):
            continue
        out.append({"video_id": e["id"], "title": e.get("title"), "duration": e.get("duration"),
                    "channel_id": e.get("channel_id"), "channel": e.get("channel") or e.get("uploader")})
    log.info("búsqueda '%s' → %d resultados", query, len(out))
    return out


def fetch_metadata(video_id: str) -> dict | None:
    """Metadatos completos incluyendo `heatmap` (lista de 100 bins con value ∈ [0,1])."""
    try:
        with _ydl({"skip_download": True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception as exc:  # videos privados, bloqueados por región, etc.
        global ultimo_error
        ultimo_error = str(exc)
        log.warning("metadatos fallidos para %s: %s", video_id, exc)
        return None
    return {
        "video_id": video_id,
        "title": info.get("title"),
        "duration": info.get("duration"),
        "channel_id": info.get("channel_id"),
        "channel": info.get("channel"),
        "language": info.get("language"),
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),
        "categories": info.get("categories"),
        "heatmap": info.get("heatmap"),
        "chapters": info.get("chapters"),
    }


def download_audio(video_id: str, out_dir: Path, fmt: str) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob(f"{video_id}.*"))
    if existing:
        return existing[0]
    try:
        with _ydl({"format": fmt, "outtmpl": str(out_dir / "%(id)s.%(ext)s")}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        return Path(ydl.prepare_filename(info))
    except Exception as exc:
        global ultimo_error
        ultimo_error = str(exc)
        log.warning("descarga de audio fallida para %s: %s", video_id, exc)
        return None


def download_video(url: str, out_dir: Path, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with _ydl({"format": fmt, "outtmpl": str(out_dir / "%(id)s.%(ext)s"), "merge_output_format": "mp4"}) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    log.info("video descargado: %s", path)
    return path


def polite_sleep(seconds: float, jitter: float = 0.0) -> None:
    """Espera `seconds` más una variación uniforme en [0, jitter] (evita un ritmo perfectamente regular)."""
    time.sleep(seconds + (random.uniform(0.0, jitter) if jitter else 0.0))
