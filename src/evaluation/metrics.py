"""Métricas de evaluación.

- Protocolo Rhapsody a nivel de bin (hit, precision, recall, F1, AP): réplica fiel de
  github.com/younghanstark/rhapsody/metrics.py (Apache-2.0) para comparabilidad con P5.
- Ranking de clips: Precision@K, MRR, NDCG@K, Spearman.
- F1 temporal vs Most Replayed: cobertura de los top-K clips (tras NMS) sobre los bins highlight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


# ---------------- Protocolo Rhapsody (listas de índices de bin) ----------------
def hit(gt: list[int], pred: list[int]) -> float:
    if len(gt) == 0:
        return float(len(pred) == 0)
    return float(len(set(pred) & set(gt)) > 0)


def precision(gt: list[int], pred: list[int]) -> float:
    if len(pred) == 0:
        return float(len(gt) == 0)
    return len(set(gt) & set(pred)) / len(pred)


def recall(gt: list[int], pred: list[int]) -> float:
    if len(gt) == 0:
        return float(len(pred) == 0)
    return len(set(gt) & set(pred)) / len(gt)


def f1(gt: list[int], pred: list[int]) -> float:
    if not gt and not pred:
        return 1.0
    if not gt or not pred or not set(gt) & set(pred):
        return 0.0
    p, r = precision(gt, pred), recall(gt, pred)
    return 2 * p * r / (p + r)


def ap(gt: list[int], pred: list[int]) -> float:
    if len(gt) == 0:
        return float(len(pred) == 0)
    s, hits = 0.0, 0
    for i, p in enumerate(pred):
        if p in gt:
            hits += 1
            s += hits / (i + 1)
    return s / len(gt)


def rhapsody_metrics(gts: list[list[int]], preds: list[list[int]]) -> dict[str, float]:
    fns = {"hit": hit, "precision": precision, "recall": recall, "f1": f1, "ap": ap}
    return {k: float(np.mean([fn(g, p) for g, p in zip(gts, preds)])) for k, fn in fns.items()}


# ---------------- Ranking de clips ----------------
def precision_at_k(rel_sorted: np.ndarray, k: int) -> float:
    return float(rel_sorted[:k].mean()) if len(rel_sorted) else 0.0


def reciprocal_rank(rel_sorted: np.ndarray) -> float:
    idx = np.nonzero(rel_sorted)[0]
    return 1.0 / (idx[0] + 1) if len(idx) else 0.0


def temporal_f1(selected: list[tuple[float, float]], gt_intervals: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Precisión = fracción del tiempo seleccionado dentro de bins highlight; recall = fracción de tiempo highlight cubierto."""
    def total(iv):
        return sum(e - s for s, e in _merge(iv))

    sel, gt = _merge(selected), _merge(gt_intervals)
    inter = sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in sel for s2, e2 in gt)
    p = inter / total(sel) if sel else 0.0
    r = inter / total(gt) if gt else 0.0
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def _merge(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def group_ranking_metrics(df: pd.DataFrame, score_col: str, ks=(1, 5, 10)) -> dict[str, float]:
    """Promedio por video de P@K, MRR, NDCG@K (relevancia graduada y_grade) y Spearman(score, y_mr)."""
    res: dict[str, list[float]] = {}
    for _, g in df.groupby("video_id"):
        g = g.sort_values(score_col, ascending=False)
        rel = g["y_relevant"].to_numpy()
        if rel.sum() == 0:
            continue
        for k in ks:
            res.setdefault(f"P@{k}", []).append(precision_at_k(rel, k))
        res.setdefault("MRR", []).append(reciprocal_rank(rel))
        if len(g) > 1:
            for k in (5, 10):
                res.setdefault(f"NDCG@{k}", []).append(
                    float(ndcg_score([g["y_grade"].to_numpy()], [g[score_col].to_numpy()], k=k)))
            rho = spearmanr(g[score_col], g["y_mr"]).statistic
            res.setdefault("Spearman", []).append(0.0 if np.isnan(rho) else float(rho))
    return {k: float(np.mean(v)) for k, v in res.items()} | {"n_videos": len(next(iter(res.values()), []))}


def temporal_f1_by_video(df: pd.DataFrame, score_col: str, bins: pd.DataFrame, k: int, nms_iou: float,
                         max_n_gt: int = 20, threshold: float = 0.5) -> dict[str, float]:
    """Selecciona top-k clips no solapados por video y mide F1 temporal contra los bins highlight."""
    from src.ranking.labels import gt_bins
    from src.segment.candidates import Window, nms

    ps, rs, fs = [], [], []
    for vid, g in df.groupby("video_id"):
        b = bins[bins["video_id"] == vid].sort_values("start")
        vals = b["y_mr"].to_numpy()
        gt = gt_bins(vals, max_n_gt, threshold)
        if not gt:
            continue
        gt_iv = [(b["start"].iloc[i], b["end"].iloc[i]) for i in gt]
        wins = [Window(s, e) for s, e in zip(g["start"], g["end"])]
        keep = nms(wins, g[score_col].to_numpy(), nms_iou, k)
        p, r, f = temporal_f1([(wins[i].start, wins[i].end) for i in keep], gt_iv)
        ps.append(p), rs.append(r), fs.append(f)
    return {"tP": float(np.mean(ps)), "tR": float(np.mean(rs)), "tF1": float(np.mean(fs)), "n_videos": len(fs)}


def bin_protocol(bins: pd.DataFrame, score_col: str, k: int, max_n_gt: int = 20, threshold: float = 0.5) -> dict[str, float]:
    """Protocolo Rhapsody: por video se predicen los top-k bins por score."""
    from src.ranking.labels import gt_bins

    gts, preds = [], []
    for _, b in bins.groupby("video_id"):
        b = b.sort_values("start").reset_index(drop=True)
        gt = gt_bins(b["y_mr"].to_numpy(), max_n_gt, threshold)
        if not gt:
            continue
        gts.append(gt)
        preds.append(b[score_col].to_numpy().argsort()[::-1][:k].tolist())
    return rhapsody_metrics(gts, preds) | {"n_videos": len(gts)}
