"""Features acústicas con librosa (técnica 8, P7 UMT): energía RMS, pitch y pausas.

Se calcula una sola vez por video a nivel de frame (hop=512 @16 kHz ≈ 32 ms) y luego
cada ventana agrega por slicing. Todo se estandariza (z-score) respecto del propio video
para que el modelo aprenda "énfasis relativo" y no el volumen absoluto de la grabación.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class AudioFrames:
    hop_s: float
    rms_z: np.ndarray        # energía estandarizada
    f0_st: np.ndarray        # pitch en semitonos (NaN si no hay voz)
    voiced: np.ndarray       # bool

    def slice(self, start: float, end: float) -> slice:
        return slice(int(start / self.hop_s), max(int(start / self.hop_s) + 1, int(end / self.hop_s)))


def compute_frames(y: np.ndarray, sr: int, hop: int = 512) -> AudioFrames:
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop, center=True)[0]
    log_rms = np.log(rms + 1e-5)
    rms_z = (log_rms - log_rms.mean()) / (log_rms.std() + 1e-8)
    f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, frame_length=2048, hop_length=hop)
    n = min(len(rms_z), len(f0))
    rms_z, f0 = rms_z[:n], f0[:n]
    voiced = (rms_z > -0.5) & (f0 > 66) & (f0 < 399)
    f0_st = np.where(voiced, 12 * np.log2(np.maximum(f0, 1e-3) / 100.0), np.nan)
    return AudioFrames(hop / sr, rms_z.astype(np.float32), f0_st.astype(np.float32), voiced)


def load_or_compute(wav: Path, cache: Path, sr: int, hop: int) -> AudioFrames:
    if cache.exists():
        d = np.load(cache)
        return AudioFrames(float(d["hop_s"]), d["rms_z"], d["f0_st"], d["voiced"])
    from src.ingest.media import load_wav

    frames = compute_frames(load_wav(wav, sr), sr, hop)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, hop_s=frames.hop_s, rms_z=frames.rms_z, f0_st=frames.f0_st, voiced=frames.voiced)
    return frames


def window_audio(frames: AudioFrames | None, start: float, end: float, hook_s: float) -> dict[str, float]:
    keys = ["a_rms_mean", "a_rms_p95", "a_rms_std", "a_pitch_std", "a_pitch_range", "a_voiced_ratio",
            "a_energy_delta", "h_rms_onset"]
    if frames is None:
        return {k: np.nan for k in keys}
    sl = frames.slice(start, end)
    rz, f0, vc = frames.rms_z[sl], frames.f0_st[sl], frames.voiced[sl]
    if len(rz) == 0:
        return {k: np.nan for k in keys}
    f0v = f0[~np.isnan(f0)]
    prev = frames.rms_z[frames.slice(max(0.0, start - 30.0), start)]
    onset = frames.rms_z[frames.slice(start, start + hook_s)]
    return {
        "a_rms_mean": float(rz.mean()),
        "a_rms_p95": float(np.percentile(rz, 95)),
        "a_rms_std": float(rz.std()),
        "a_pitch_std": float(f0v.std()) if len(f0v) > 10 else np.nan,
        "a_pitch_range": float(np.percentile(f0v, 95) - np.percentile(f0v, 5)) if len(f0v) > 10 else np.nan,
        "a_voiced_ratio": float(vc.mean()),
        "a_energy_delta": float(rz.mean() - prev.mean()) if len(prev) else 0.0,
        "h_rms_onset": float(onset.mean()) if len(onset) else np.nan,
    }
