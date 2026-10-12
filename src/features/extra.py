"""Features de la iteración 2 (D-024), respaldadas por la segunda búsqueda de papers.

Grupo E (prefijo `e_`) — eventos de audio con PANNs CNN14 (Kong et al., TASLP 2020):
  risa, aplausos, vítores, música y habla. La risa del público como marca de momento
  enganchador viene de UR-FUNNY (Hasan et al., EMNLP 2019) y StandUp4AI (Barriere et al., 2025).
  Costo medido en el equipo del autor: RTF≈0.12 con 1 s por ventana → se muestrea 1 s cada
  `stride_s` segundos (por defecto 3 s, RTF≈0.04).

Grupo L (prefijo `l_`) — rasgos lingüísticos de engagement de Reddy et al. (ACL 2021):
  entropía léxica, ratio de pronombres 1ª/2ª vs 3ª persona, interjecciones/muletillas,
  proporción de silencio, longitud media de palabra (proxy de legibilidad).

Se calculan como columnas adicionales sobre las tablas ya featurizadas (no requiere re-featurizar).
"""
from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger
from src.keywords.text import FILLERS, FIRST_PERSON, SECOND_PERSON, lang_key, tokenize
from src.transcribe.whisper import Transcript

log = get_logger(__name__)

# índices AudioSet (class_labels_indices.csv)
EVENTS = {"speech": 0, "laughter": 16, "belly_laugh": 20, "cheering": 66, "applause": 67, "music": 137}
THIRD_PERSON = {"es": {"él", "ella", "ellos", "ellas", "le", "les", "lo", "la", "su", "sus"},
                "en": {"he", "she", "they", "him", "her", "them", "his", "their"}}


@lru_cache(maxsize=1)
def _panns(checkpoint: str, device: str = "auto"):
    from panns_inference import AudioTagging

    from src.common.devices import resolve_device

    dev = resolve_device(device)
    log.info("cargando PANNs CNN14 (device=%s)", dev)
    return AudioTagging(checkpoint_path=checkpoint, device=dev)


def audio_events(wav: Path, cache: Path, checkpoint: str, stride_s: float = 3.0,
                 device: str = "auto") -> pd.DataFrame:
    """DataFrame (t, speech, laughter, …) con probabilidades AudioSet de 1 s muestreadas cada stride_s."""
    if cache.exists():
        return pd.read_parquet(cache)
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(wav), dtype="float32")
    y32 = librosa.resample(y, orig_sr=sr, target_sr=32000)
    starts = np.arange(0, len(y32) / 32000 - 1, stride_s)
    frames = np.stack([y32[int(s * 32000): int(s * 32000) + 32000] for s in starts])
    at = _panns(checkpoint, device)
    probs = []
    for i in range(0, len(frames), 64):
        cw, _ = at.inference(frames[i:i + 64])
        probs.append(cw)
    p = np.concatenate(probs)
    df = pd.DataFrame({"t": starts + 0.5, **{k: p[:, j] for k, j in EVENTS.items()}})
    df["laugh"] = np.maximum(df["laughter"], df["belly_laugh"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def event_features(ev: pd.DataFrame, start: float, end: float) -> dict[str, float]:
    t = ev["t"].to_numpy()
    m = (t >= start) & (t < end)
    after = (t >= end) & (t < end + 6)  # reacción inmediatamente posterior (risa/aplauso al remate)
    tail = (t >= end - 6) & (t < end)
    out = {}
    for k in ("laugh", "applause", "cheering", "music", "speech"):
        v = ev[k].to_numpy()
        vm = v[m]
        out[f"e_{k}_max"] = float(vm.max()) if vm.size else np.nan
        out[f"e_{k}_mean"] = float(vm.mean()) if vm.size else np.nan
    lv = ev["laugh"].to_numpy()
    out["e_laugh_frac"] = float((lv[m] > 0.2).mean()) if m.any() else np.nan
    out["e_laugh_tail"] = float(lv[tail].max()) if tail.any() else np.nan
    out["e_laugh_after"] = float(lv[after].max()) if after.any() else np.nan
    out["e_applause_after"] = float(ev["applause"].to_numpy()[after].max()) if after.any() else np.nan
    return out


def linguistic_features(tr: Transcript, start: float, end: float) -> dict[str, float]:
    lang = lang_key(tr.language)
    words = [w for w in tr.words if start <= w.start < end]
    toks = [t for w in words for t in tokenize(w.text)]
    n = len(toks)
    out: dict[str, float] = {}
    if n == 0:
        return {k: np.nan for k in ("l_entropy", "l_type_token", "l_pron_12_vs_3", "l_fillers", "l_silence",
                                    "l_word_len")}
    c = Counter(toks)
    out["l_entropy"] = -sum((v / n) * math.log2(v / n) for v in c.values())
    out["l_type_token"] = len(c) / n
    p12 = sum(c[t] for t in FIRST_PERSON[lang] | SECOND_PERSON[lang])
    p3 = sum(c[t] for t in THIRD_PERSON[lang])
    out["l_pron_12_vs_3"] = (p12 + 1) / (p3 + 1)
    out["l_fillers"] = sum(c[t] for t in FILLERS[lang] if " " not in t) / n
    spoken = sum(w.end - w.start for w in words)
    out["l_silence"] = 1 - spoken / max(end - start, 1e-3)
    out["l_word_len"] = float(np.mean([len(t) for t in toks]))
    return out


def augment_table(df: pd.DataFrame, tr: Transcript, ev: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for s, e in zip(df["start"], df["end"]):
        r = linguistic_features(tr, s, e)
        if ev is not None:
            r.update(event_features(ev, s, e))
        rows.append(r)
    extra = pd.DataFrame(rows, index=df.index)
    keep = df.drop(columns=[c for c in df.columns if c.startswith(("e_", "l_"))])
    return pd.concat([keep, extra], axis=1)
