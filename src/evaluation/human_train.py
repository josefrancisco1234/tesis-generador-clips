"""Entrenamiento con la etiqueta de curación humana (Repurpose-10K) y comparación con Most Replayed.

Responde la pregunta que abrió la iteración 2 (D-028): si la curva de re-reproducción no predice
lo que eligen los editores, ¿un modelo entrenado directamente con clips humanos sí lo hace?

Requiere al menos `--min-videos` videos con `y_human`. Validación cruzada agrupada por canal.

    python -m src.evaluation.human_train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

from src.common.config import load_config
from src.common.logging_utils import get_logger, setup_logging
from src.data.featurize import load_table
from src.evaluation.baselines import random_scores
from src.evaluation.experiments import inner_val, lgbm_predict, sub
from src.evaluation.metrics import precision_at_k, reciprocal_rank, temporal_f1
from src.features.builder import feature_columns
from src.ranking.labels import grade_within
from src.ranking.model import train_lgbm
from src.segment.candidates import Window, nms

log = get_logger(__name__)


def per_video_human(df: pd.DataFrame, col: str, segments: dict[str, list], k: int, overlap: float) -> pd.DataFrame:
    rows = []
    for vid, g in df.groupby("video_id"):
        g = g.reset_index(drop=True)
        wins = [Window(s, e) for s, e in zip(g["start"], g["end"])]
        keep = nms(wins, g[col].to_numpy(), overlap, k, mode="iomin")
        order = g.sort_values(col, ascending=False)["y_human"].to_numpy()
        _, _, tf = temporal_f1([(wins[i].start, wins[i].end) for i in keep], segments[vid])
        rows.append({"video_id": vid, f"P@{k}": float(g.iloc[keep]["y_human"].mean()),
                     "MRR": reciprocal_rank(order), "P@1": precision_at_k(order, 1), "tF1": tf})
    return pd.DataFrame(rows).set_index("video_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-videos", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="reports/entrenamiento_humano.json")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging("WARNING")
    from src.common.io import read_json
    from src.data.repurpose import clips_path

    cand = load_table(cfg, "sentence")
    if "y_human" not in cand.columns:
        raise SystemExit("no hay columna y_human: ejecuta primero python -m src.data.augment")
    cand = cand[cand["y_human"].notna()].reset_index(drop=True)
    videos = cand.groupby("video_id").agg(channel_id=("channel_id", "first")).reset_index()
    n = len(videos)
    print(f"videos con curación humana: {n}")
    if n < args.min_videos:
        raise SystemExit(f"hacen falta al menos {args.min_videos} videos; hay {n}. "
                         "Sigue descargando Repurpose-10K y vuelve a ejecutar.")

    segments = {v: [(s, e) for s, e in read_json(clips_path(cfg, v))["segments"]] for v in videos["video_id"]}
    cand["y_grade_human"] = cand.groupby("video_id")["y_human_overlap"].transform(
        lambda s: grade_within(s, cfg.labels["n_grades"]))
    feats = feature_columns(cand, cfg.features["groups"])
    folds = min(args.folds, n)
    preds = []
    for fi, (a, t) in enumerate(GroupKFold(folds).split(videos, groups=videos["channel_id"].fillna(videos["video_id"]))):
        test_ids = videos.iloc[t]["video_id"].tolist()
        tr_ids, va_ids = inner_val(videos, videos.iloc[a]["video_id"].tolist(), cfg.seed + fi)
        te = sub(cand, test_ids)
        for name, label in (("entrenado_con_humano", "y_grade_human"), ("entrenado_con_mostreplayed", "y_grade")):
            b = train_lgbm(sub(cand, tr_ids), sub(cand, va_ids), feats, cfg.ranking["lgbm"], cfg.seed, label)
            te[name] = lgbm_predict(b, te, feats)
        te["aleatorio"] = random_scores(te, cfg.seed)
        preds.append(te)
    pred = pd.concat(preds, ignore_index=True)

    res, out = {}, {"n_videos": n}
    for m in ("entrenado_con_humano", "entrenado_con_mostreplayed", "aleatorio"):
        res[m] = per_video_human(pred, m, segments, args.k, cfg.ranking["output_overlap"])
    tabla = pd.DataFrame({m: v.mean() for m, v in res.items()}).T
    print("\n== Recuperación de clips humanos (validación cruzada) ==")
    print(tabla.round(3).to_string())
    out["metricas"] = tabla.reset_index().rename(columns={"index": "modelo"}).to_dict("records")
    out["tests"] = {}
    for m in ("entrenado_con_humano", "entrenado_con_mostreplayed"):
        d = (res[m]["tF1"] - res["aleatorio"]["tF1"]).dropna().to_numpy()
        p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        out["tests"][f"{m}_vs_aleatorio_tF1"] = p
    d = (res["entrenado_con_humano"]["tF1"] - res["entrenado_con_mostreplayed"]["tF1"]).dropna().to_numpy()
    out["tests"]["humano_vs_mostreplayed_tF1"] = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    print("\np (Wilcoxon unilateral):", {k: round(v, 4) for k, v in out["tests"].items()})
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
