"""Features relativas (iteración 2, D-022).

La curva Most Replayed mide *picos* de re-reproducción respecto del resto del video, no valores
absolutos. Se derivan dos familias a partir de las features base, conservando el prefijo de grupo
(k_, s_, a_, h_, f_) para que ablaciones y desglose Hook/Flow/Value sigan funcionando:

  <feat>__rk : percentil dentro del video (0–1)
  <feat>__cx : contraste local = valor − media de los candidatos cuyo centro está a ≤ `radius_s`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_PREFIXES = ("k_", "s_", "a_", "h_", "f_", "e_", "l_")  # u_ queda fuera: ya son componentes PCA


def base_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(BASE_PREFIXES) and "__" not in c]


def add_relative(df: pd.DataFrame, radius_s: float = 90.0, cols: list[str] | None = None,
                 kinds: tuple[str, ...] = ("rk", "cx")) -> pd.DataFrame:
    cols = cols or [c for c in base_features(df) if df[c].nunique(dropna=True) > 2]
    out = df.copy()
    new: dict[str, np.ndarray] = {}
    if "rk" in kinds:
        ranks = df.groupby("video_id")[cols].rank(pct=True)
        for c in cols:
            new[f"{c}__rk"] = ranks[c].to_numpy()
    if "cx" in kinds:
        cx = np.full((len(df), len(cols)), np.nan)
        centers = ((df["start"] + df["end"]) / 2).to_numpy()
        vals = df[cols].to_numpy(dtype=float)
        for _, idx in df.groupby("video_id").indices.items():
            order = idx[np.argsort(centers[idx])]
            c_sorted = centers[order]
            lo = np.searchsorted(c_sorted, c_sorted - radius_s, "left")
            hi = np.searchsorted(c_sorted, c_sorted + radius_s, "right")
            v = vals[order]
            filled = np.where(np.isnan(v), 0.0, v)
            cnt = np.cumsum(np.vstack([np.zeros((1, v.shape[1])), ~np.isnan(v)]), axis=0)
            csum = np.cumsum(np.vstack([np.zeros((1, v.shape[1])), filled]), axis=0)
            local_mean = (csum[hi] - csum[lo]) / np.maximum(cnt[hi] - cnt[lo], 1)
            cx[order] = v - local_mean
        for j, c in enumerate(cols):
            new[f"{c}__cx"] = cx[:, j]
    return pd.concat([out, pd.DataFrame(new, index=df.index)], axis=1)
