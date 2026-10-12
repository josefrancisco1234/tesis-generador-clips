"""Operaciones con ffmpeg/ffprobe."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger(__name__)


def _require(tool: str) -> str:
    exe = shutil.which(tool)
    if not exe:
        raise RuntimeError(f"{tool} no está en el PATH")
    return exe


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    log.debug("ejecutando: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"falló {cmd[0]}: {proc.stderr[-1500:]}")
    return proc


def probe(path: Path) -> dict:
    out = run([_require("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    info = json.loads(out.stdout)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    res = {"duration": float(info["format"]["duration"]), "has_video": video is not None}
    if video:
        num, den = (video.get("avg_frame_rate") or "30/1").split("/")
        res.update(width=int(video["width"]), height=int(video["height"]),
                   fps=float(num) / float(den) if float(den) else 30.0)
    return res


def extract_audio(src: Path, dst: Path, sr: int = 16000) -> Path:
    """Audio mono PCM 16 bits a `sr` Hz (entrada común para whisper y librosa)."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([_require("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
         "-c:a", "pcm_s16le", str(dst)])
    return dst


def load_wav(path: Path, sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    y, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y
