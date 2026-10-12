"""Evaluación contra los clips elegidos por humanos (Repurpose-10K, D-023).

Dos preguntas:
  1. ¿Coinciden la re-reproducción del público (Most Replayed) y la curación humana?
  2. ¿El ranker recupera los clips humanos, aunque nunca se entrenó con ellos?

Usa las predicciones fuera de fold (`reports/cv_predictions_cand.parquet`), así ningún video
se evalúa con un modelo que lo haya visto en entrenamiento.

    python -m src.evaluation.human_clips
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.common.config import load_config
from src.common.io import read_json
from src.common.logging_utils import setup_logging
from src.data.repurpose import clips_path
from src.evaluation.metrics import precision_at_k, reciprocal_rank, temporal_f1
from src.ranking.labels import corrected_graph, gt_bins
from src.segment.candidates import Window, nms

MODELS = ["lgbm_autogroups", "lgbm_full", "opus_heuristic", "keybert_only", "random"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="reports/clips_humanos.json")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging("WARNING")
    cand = pd.read_parquet(cfg.path("reports") / "cv_predictions_cand.parquet")
    if "y_human" not in cand.columns:
        raise SystemExit("las predicciones no tienen y_human: reejecuta run_all.py --only experiments")
    vids = sorted(cand.loc[cand["y_human"].notna(), "video_id"].unique())
    print(f"videos con clips humanos: {len(vids)}")

    agree, rows = [], []
    for vid in vids:
        g = cand[cand["video_id"] == vid].reset_index(drop=True)
        segs = read_json(clips_path(cfg, vid))["segments"]
        meta = read_json(cfg.path("interim") / "heatmaps" / f"{vid}.json")
        bs, be, bv = corrected_graph(meta["heatmap"])
        gt = gt_bins(bv, cfg.labels["max_n_gt"], cfg.labels["gt_threshold"])
        # 1) acuerdo entre las dos etiquetas
        rho = spearmanr(g["y_mr"], g["y_human_overlap"]).statistic
        p, r, f = temporal_f1([(bs[i], be[i]) for i in gt], [(s, e) for s, e in segs])
        agree.append({"video_id": vid, "spearman_mr_humano": float(rho) if not np.isnan(rho) else 0.0,
                      "f1_temporal_mr_vs_humano": f, "n_clips_humanos": len(segs),
                      "frac_candidatos_humanos": float(g["y_human"].mean())})
        # 2) ¿el ranker recupera los clips humanos?
        wins = [Window(s, e) for s, e in zip(g["start"], g["end"])]
        for m in MODELS:
            if m not in g:
                continue
            keep = nms(wins, g[m].to_numpy(), cfg.ranking["output_overlap"], args.k, mode="iomin")
            sel = g.iloc[keep]
            _, _, tf = temporal_f1([(wins[i].start, wins[i].end) for i in keep], [(s, e) for s, e in segs])
            order = g.sort_values(m, ascending=False)
            rows.append({"video_id": vid, "modelo": m, f"P@{args.k}": float(sel["y_human"].mean()),
                         "MRR": reciprocal_rank(order["y_human"].to_numpy()),
                         "P@1": precision_at_k(order["y_human"].to_numpy(), 1),
                         "tF1_vs_humano": tf})

    ag = pd.DataFrame(agree)
    res = pd.DataFrame(rows)
    piv = res.groupby("modelo")[[f"P@{args.k}", "P@1", "MRR", "tF1_vs_humano"]].mean().sort_values("tF1_vs_humano", ascending=False)
    print("\n== Acuerdo entre Most Replayed y curación humana ==")
    print(ag[["spearman_mr_humano", "f1_temporal_mr_vs_humano", "frac_candidatos_humanos"]].describe().loc[
        ["mean", "50%", "min", "max"]].round(3).to_string())
    print("\n== ¿El ranker recupera los clips humanos? (predicciones fuera de fold) ==")
    print(piv.round(3).to_string())
    tests = {}
    for m in [x for x in MODELS if x != "random" and x in res["modelo"].values]:
        a = res[res["modelo"] == m].set_index("video_id")["tF1_vs_humano"]
        b = res[res["modelo"] == "random"].set_index("video_id")["tF1_vs_humano"]
        d = (a - b).dropna().to_numpy()
        tests[m] = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    print("\np (Wilcoxon unilateral vs azar, tF1 contra clips humanos):", {k: round(v, 4) for k, v in tests.items()})
    out = {"n_videos": len(vids), "acuerdo": ag.to_dict("records"),
           "ranking": piv.reset_index().to_dict("records"), "p_vs_random": tests}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
