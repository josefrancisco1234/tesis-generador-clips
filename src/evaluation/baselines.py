"""Líneas base de comparación (todas producen una columna de score por ventana)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_scores(df: pd.DataFrame, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(len(df))


def position_prior(train: pd.DataFrame, test: pd.DataFrame, n_bins: int = 20) -> np.ndarray:
    """Equivalente al baseline 'frequency' de Rhapsody: y_mr medio por posición relativa en train."""
    edges = np.linspace(0, 1, n_bins + 1)
    tb = np.clip(np.digitize(train["p_rel_pos"], edges) - 1, 0, n_bins - 1)
    prior = pd.Series(train["y_mr"].to_numpy()).groupby(tb).mean().reindex(range(n_bins)).fillna(train["y_mr"].mean())
    qb = np.clip(np.digitize(test["p_rel_pos"], edges) - 1, 0, n_bins - 1)
    return prior.to_numpy()[qb]


def single_feature(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].fillna(df[col].median() if df[col].notna().any() else 0).to_numpy()


# Heurística tipo Opus (Hook/Flow/Value con pesos fijos, sin aprendizaje): D-006
OPUS_HEURISTIC = {
    "h_question": 1.0, "h_hook_word": 1.0, "h_second_person": 0.5, "h_number": 0.5, "h_rms_onset": 0.5,
    "f_ends_complete": 1.0, "f_starts_dangling": -1.0, "f_coherence_mean": 1.0,
    "k_kvsum_local": 1.0, "k_keybert_sim": 1.0, "s_intensity_mean": 1.0, "a_rms_mean": 0.5, "a_pitch_std": 0.5,
}


def opus_heuristic(df: pd.DataFrame) -> np.ndarray:
    score = np.zeros(len(df))
    for col, w in OPUS_HEURISTIC.items():
        if col not in df:
            continue
        x = df[col].astype(float)
        # z-score dentro de cada video para que los pesos sean comparables
        z = x.groupby(df["video_id"]).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8)).fillna(0)
        score += w * z.to_numpy()
    return score
