"""Seleccion de clips guiada por retencion (D-045 / D-047).

Dos caminos, ambos con filtro de calidad editorial (inicio no colgado y final completo):
  - CALOR: si el video de YouTube trae la curva Most Replayed, el score de cada candidato es el
    calor medio de su ventana (confirmado: F1 temporal 0.47, clips ~3x mas "calientes").
  - FORMULA: si no hay curva, suma ponderada de rangos percentiles de features de contenido con
    pesos congelados de D-045 (confirmado en 352 videos nuevos: F1 0.13 vs 0.07 del azar).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

log = get_logger(__name__)
FORMULA_FILE = "formula_calor_D045.json"
_YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def load_formula(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / FORMULA_FILE).read_text(encoding="utf-8"))


def formula_raw(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Suma ponderada de rangos percentiles DENTRO del video (columna ausente o NaN -> 0.5)."""
    cols = []
    for c in weights:
        s = df[c] if c in df else pd.Series(np.nan, index=df.index)
        cols.append(s.rank(pct=True).fillna(0.5).to_numpy())
    return np.column_stack(cols) @ np.array(list(weights.values()), dtype=float)


def quality_ok(df: pd.DataFrame) -> np.ndarray:
    """Candidato autocontenido: no empieza colgado ("eso", "él"...) y termina en oración completa."""
    return ((df["f_starts_dangling"] == 0) & (df["f_ends_complete"] == 1)).to_numpy()


def calibrate(raw: np.ndarray, table: list[float]) -> np.ndarray:
    """Percentil 0..99 del score frente a los top-10 por video del conjunto de desarrollo."""
    return np.clip(np.searchsorted(np.asarray(table), raw, side="right") - 1, 0, 99).astype(int)


def video_id_from_url(src: str) -> str | None:
    m = _YT_ID.search(src or "")
    return m.group(1) if m else None


def heat_raw(df: pd.DataFrame, heatmap: list[dict]) -> np.ndarray:
    """Calor medio (curva corregida, bin 0 = bin 1) de cada ventana candidata."""
    from src.ranking.labels import corrected_graph, window_mr
    from src.segment.candidates import Window

    st, en, vals = corrected_graph(heatmap)
    out = [window_mr(Window(float(a), float(b), 0, 0), st, en, vals) for a, b in zip(df["start"], df["end"])]
    return np.nan_to_num(np.asarray(out, dtype=float), nan=0.0)


def apply(df: pd.DataFrame, cfg, src: str, method: str, model_name: str,
          max_clip_s: float | None = None) -> tuple[pd.DataFrame, str]:
    """Reemplaza `raw` y `score` segun el metodo; deja intacto el desglose Hook/Flow/Value."""
    heatmap = None
    if method in ("auto", "heat"):
        vid = video_id_from_url(src)
        if vid:
            from src.ingest import youtube

            meta = youtube.fetch_metadata(vid)
            heatmap = (meta or {}).get("heatmap")
        if not heatmap:
            log.warning("sin curva Most Replayed para %r; uso la formula de contenido", src)
    df = df.copy()
    ok = quality_ok(df)
    if heatmap:
        base = heat_raw(df, heatmap)
        score = np.round(np.clip(base / max(float(base.max()), 1e-9), 0.0, 1.0) * 99).astype(int)
        label = "calor + calidad (curva Most Replayed de YouTube)"
    else:
        formula = load_formula(cfg.path("models"))
        base = formula_raw(df, formula["weights"])
        score = calibrate(base, formula["calibrator"])
        label = "formula de contenido + calidad (D-045/D-047)"
        if max_clip_s:   # V1 (D-048/D-049): formula + solo clips cortos, tal como se confirmo (sin filtro de calidad)
            ok = ((df["end"] - df["start"]).to_numpy() <= float(max_clip_s))
            label = f"formula de contenido + clips <= {max_clip_s:g} s (V1, D-049)"
    df["raw"] = np.where(ok, base, float(base.min()) - 10.0)   # los no autocontenidos quedan al final
    df["score"] = score
    for d in ("hook", "flow", "value"):
        if f"score_{d}" not in df:
            df[f"score_{d}"] = -1
    log.info("metodo de seleccion: %s (%d/%d candidatos autocontenidos)", label, int(ok.sum()), len(df))
    return df, label


def explain(df: pd.DataFrame, i: int, weights: dict[str, float]) -> dict:
    """Por que la ventana i puntua alto: valor, percentil dentro del video, peso y aporte de cada variable de la formula."""
    rows = []
    for k, w in weights.items():
        if k in df and df[k].notna().any():
            pct = float(df[k].rank(pct=True).fillna(0.5).iloc[i])
            val = df[k].iloc[i]
            val = None if pd.isna(val) else float(val)
        else:
            pct, val = 0.5, None
        rows.append({"var": k, "value": val, "pct": round(pct, 3), "weight": round(w, 4), "contrib": round(w * pct, 4)})
    rows.sort(key=lambda r: -abs(r["contrib"]))
    raw = df["raw"].to_numpy()
    return {"vars": rows, "rank": int((raw > raw[i]).sum()) + 1, "n_candidates": int(len(raw))}
