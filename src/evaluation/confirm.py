"""Prueba de confirmación de la hipótesis preregistrada D-022.

Entrena en los videos de DESARROLLO (v1) y evalúa en los videos NUEVOS (v2), que no
participaron en ninguna decisión. Compara `base` vs `rel_rk` con Wilcoxon pareado por video,
unilateral (H1: rel_rk > base en NDCG@10), α = 0.05.

    python -m src.evaluation.confirm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.common.config import load_config
from src.common.logging_utils import get_logger, setup_logging
from src.data.featurize import load_table
from src.evaluation.experiments import inner_val, lgbm_predict, sub
from src.evaluation.iterate import make_variant
from src.evaluation.significance import per_video
from src.ranking.model import train_lgbm

log = get_logger(__name__)
PRIMARY = "NDCG@10"
SECONDARY = ["tF1@5", "P@5", "MRR", "bin_f1@5"]


def valid_ids(cfg) -> pd.DataFrame:
    base = cfg.path("manifests")
    parts = [pd.read_csv(base / f"{n}_status.csv") for n in ("youtube_mr", "youtube_mr_repurpose")
             if (base / f"{n}_status.csv").exists()]
    st = pd.concat(parts)
    return st[st["ok"] == True]  # noqa: E712


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/confirmacion_D022.json")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging("WARNING")
    st = valid_ids(cfg)
    dev_ids = pd.read_csv(cfg.path("manifests") / "youtube_mr_status_v1.csv").query("ok == True")["video_id"].tolist()
    dev = [v for v in st["video_id"] if v in set(dev_ids)]
    new = [v for v in st["video_id"] if v not in set(dev_ids)]
    cand_all, bins_all = load_table(cfg, "sentence"), load_table(cfg, "bin")
    dev = [v for v in dev if v in set(cand_all["video_id"])]
    new = [v for v in new if v in set(cand_all["video_id"])]
    videos = cand_all.groupby("video_id").agg(channel_id=("channel_id", "first")).reset_index()
    print(f"desarrollo: {len(dev)} videos · confirmación (nuevos): {len(new)} videos")

    results, pvs = {}, {}
    for name in ("base", "rel_rk"):
        c, b, feats, label = make_variant(name, cand_all, bins_all, cfg)
        tr_ids, va_ids = inner_val(videos[videos["video_id"].isin(dev)], dev, cfg.seed)
        booster = train_lgbm(sub(c, tr_ids), sub(c, va_ids), feats, cfg.ranking["lgbm"], cfg.seed, label)
        tc, tb = sub(c, new), sub(b, new)
        tc[name], tb[name] = lgbm_predict(booster, tc, feats), lgbm_predict(booster, tb, feats)
        pvs[name] = per_video(tc, tb, name)
        results[name] = {m: float(pvs[name][m].mean()) for m in [PRIMARY] + SECONDARY}

    rows, out = [], {"n_dev": len(dev), "n_new": len(new), "metrics": results, "tests": {}}
    for m in [PRIMARY] + SECONDARY:
        a, b = pvs["rel_rk"][m], pvs["base"][m]
        ok = a.notna() & b.notna()
        d = (a[ok] - b[ok]).to_numpy()
        p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        out["tests"][m] = {"n": int(ok.sum()), "base": float(b[ok].mean()), "rel_rk": float(a[ok].mean()),
                           "dif": float(d.mean()), "p_unilateral": p,
                           "significativo": bool(p < 0.05) if m == PRIMARY else None}
        rows.append({"métrica": m, "base": b[ok].mean(), "rel_rk": a[ok].mean(), "dif": d.mean(), "p": p})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    verdict = "CONFIRMADA" if out["tests"][PRIMARY]["p_unilateral"] < 0.05 else "NO CONFIRMADA"
    out["veredicto"] = verdict
    print(f"\nHipótesis D-022 ({PRIMARY}, unilateral, α=0.05): {verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
