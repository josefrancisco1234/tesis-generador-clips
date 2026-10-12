"""Pruebas de significancia pareadas por video (Wilcoxon signed-rank) sobre las predicciones de CV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.evaluation.metrics import bin_protocol, group_ranking_metrics, temporal_f1_by_video

MODELS = ["lgbm_autogroups", "lgbm_autogroups_U", "lgbm_full_U", "only_U", "lgbm_full", "mlp_ranknet", "opus_heuristic", "keybert_only", "tfidf_only", "audio_rms_only",
          "position_prior", "random", "ablate_-K", "ablate_-S", "ablate_-A", "ablate_-H", "ablate_-F", "ablate_+P"]


def per_video(cand: pd.DataFrame, bins: pd.DataFrame, model: str, nms_iou: float = 0.5) -> pd.DataFrame:
    rows = []
    for vid, g in cand.groupby("video_id"):
        b = bins[bins["video_id"] == vid]
        r = {"video_id": vid}
        m = group_ranking_metrics(g, model)
        r.update({k: m.get(k, np.nan) for k in ("P@5", "MRR", "NDCG@10")})
        r["tF1@5"] = temporal_f1_by_video(g, model, b, 5, nms_iou)["tF1"]
        r["bin_f1@5"] = bin_protocol(b, model, 5)["f1"] if model in b else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("video_id")


def run(reports: Path, reference: str = "lgbm_full") -> pd.DataFrame:
    cand = pd.read_parquet(reports / "cv_predictions_cand.parquet")
    bins = pd.read_parquet(reports / "cv_predictions_bins.parquet")
    models = [m for m in MODELS if m in cand.columns]
    pv = {m: per_video(cand, bins, m) for m in models}
    rows = []
    for m in models:
        if m == reference:
            continue
        for metric in ("P@5", "MRR", "NDCG@10", "tF1@5", "bin_f1@5"):
            a, b = pv[reference][metric], pv[m][metric]
            ok = a.notna() & b.notna()
            d = (a[ok] - b[ok]).to_numpy()
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            rows.append({"comparación": f"{reference} vs {m}", "métrica": metric, "n_videos": int(ok.sum()),
                         "media_ref": a[ok].mean(), "media_otro": b[ok].mean(), "diferencia": d.mean(), "p_wilcoxon": p})
    out = pd.DataFrame(rows)
    out.to_csv(reports / "significance.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse

    from src.common.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--reference", default="lgbm_full",
                    help="modelo de referencia; D-038 compara lgbm_autogroups_U contra lgbm_autogroups")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    print(run(load_config(a.config).path("reports"), a.reference).round(4).to_string())
