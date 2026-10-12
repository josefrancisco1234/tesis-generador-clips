"""Etiquetas débiles desde la curva Most Replayed (técnica 3; P3 SPOT, P5 Rhapsody)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.segment.candidates import Window


def corrected_graph(heatmap: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(starts, ends, values). Corrige el sesgo del primer bin como Rhapsody (graph[0] = graph[1])."""
    starts = np.array([b["start_time"] for b in heatmap], dtype=float)
    ends = np.array([b["end_time"] for b in heatmap], dtype=float)
    vals = np.array([b["value"] for b in heatmap], dtype=float)
    if len(vals) > 1:
        vals = vals.copy()
        vals[0] = vals[1]
    return starts, ends, vals


def gt_bins(values: np.ndarray, max_n: int = 20, threshold: float = 0.5) -> list[int]:
    """Bins highlight según Rhapsody: score > umbral, ordenados, máximo max_n."""
    idx = np.nonzero(values > threshold)[0]
    return idx[np.argsort(values[idx])[::-1]][:max_n].tolist()


def _overlaps(w: Window, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return np.clip(np.minimum(ends, w.end) - np.maximum(starts, w.start), 0, None)


def window_mr(w: Window, starts: np.ndarray, ends: np.ndarray, vals: np.ndarray) -> float:
    ov = _overlaps(w, starts, ends)
    return float((ov * vals).sum() / ov.sum()) if ov.sum() > 0 else np.nan


def window_gt_overlap(w: Window, starts: np.ndarray, ends: np.ndarray, gt: list[int]) -> float:
    ov = _overlaps(w, starts, ends)
    total = w.end - w.start
    return float(ov[gt].sum() / total) if total > 0 and gt else 0.0


def add_labels(df: pd.DataFrame, windows: list[Window], heatmap: list[dict], max_n: int, threshold: float,
               n_grades: int, relevant_overlap: float) -> pd.DataFrame:
    starts, ends, vals = corrected_graph(heatmap)
    gt = gt_bins(vals, max_n, threshold)
    df = df.copy()
    df["y_mr"] = [window_mr(w, starts, ends, vals) for w in windows]
    df["y_gt_overlap"] = [window_gt_overlap(w, starts, ends, gt) for w in windows]
    df["y_relevant"] = (df["y_gt_overlap"] >= relevant_overlap).astype(int)
    df["y_grade"] = grade_within(df["y_mr"], n_grades)
    return df


def grade_within(y: pd.Series, n_grades: int) -> pd.Series:
    """Relevancia graduada 0..n-1 por cuantiles dentro de un mismo video (lo que espera LambdaRank)."""
    r = (y.rank(method="first") - 1) / max(y.notna().sum(), 1)  # 0 ≤ r < 1
    return np.minimum((r * n_grades).fillna(0).astype(int), n_grades - 1).astype(int)
