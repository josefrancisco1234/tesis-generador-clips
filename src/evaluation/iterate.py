"""Comparación rápida de variantes (features / etiquetas) con CV agrupada por canal.

Metodología de la iteración 2 (D-021): las variantes se comparan SOLO sobre los videos de v1
(desarrollo). La variante elegida se confirma después sobre los videos nuevos de v2, que no
participan en ninguna decisión (test de confirmación).

    python -m src.evaluation.iterate --variants base,rel_rk,rel_cx,rel_both,label_peak
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

from src.common.config import Config, load_config, set_seeds
from src.common.logging_utils import get_logger, setup_logging
from src.data.featurize import load_table
from src.evaluation.experiments import inner_val, lgbm_predict, sub
from src.evaluation.significance import per_video
from src.features.builder import feature_columns
from src.features.relative import add_relative
from src.ranking.labels import grade_within
from src.ranking.model import train_lgbm

log = get_logger(__name__)


def window_peak(cand: pd.DataFrame, bins: pd.DataFrame) -> np.ndarray:
    """Máximo de la curva (corregida) entre los bins que se solapan con cada candidato."""
    out = np.full(len(cand), np.nan)
    for vid, idx in cand.groupby("video_id").indices.items():
        b = bins[bins["video_id"] == vid].sort_values("start")
        bs, be, bv = b["start"].to_numpy(), b["end"].to_numpy(), b["y_mr"].to_numpy()
        for i in idx:
            s, e = cand.at[i, "start"], cand.at[i, "end"]
            m = (be > s) & (bs < e)
            out[i] = bv[m].max() if m.any() else np.nan
    return out


def make_variant(name: str, cand: pd.DataFrame, bins: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """Devuelve (candidatos, bins, features, columna de etiqueta) para la variante."""
    groups = cfg.features["groups"]
    c, b, label = cand, bins, "y_grade"
    if name.startswith("rel"):
        kinds = {"rel_rk": ("rk",), "rel_cx": ("cx",), "rel_both": ("rk", "cx")}[name.split("+")[0]]
        c, b = add_relative(cand, kinds=kinds), add_relative(bins, kinds=kinds)
    if "label_peak" in name:
        c = c.assign(y_peak=window_peak(c, bins))
        c["y_grade_peak"] = c.groupby("video_id")["y_peak"].transform(lambda s: grade_within(s, cfg.labels["n_grades"]))
        label = "y_grade_peak"
    if "label_g10" in name:
        c = c.assign(y_grade10=c.groupby("video_id")["y_mr"].transform(lambda s: grade_within(s, 10)))
        label = "y_grade10"
    feats = feature_columns(c, groups)
    return c, b, feats, label


def cv_variant(name: str, cand: pd.DataFrame, bins: pd.DataFrame, videos: pd.DataFrame, cfg: Config, folds: int = 5) -> pd.DataFrame:
    c, b, feats, label = make_variant(name, cand, bins, cfg)
    params = dict(cfg.ranking["lgbm"])
    if label == "y_grade10":
        params["label_gain"] = list(range(10))
    preds_c, preds_b = [], []
    for fi, (a, t) in enumerate(GroupKFold(folds).split(videos, groups=videos["channel_id"].fillna(videos["video_id"]))):
        test_ids = videos.iloc[t]["video_id"].tolist()
        tr_ids, va_ids = inner_val(videos, videos.iloc[a]["video_id"].tolist(), cfg.seed + fi)
        booster = train_lgbm(sub(c, tr_ids), sub(c, va_ids), feats, params, cfg.seed, label)
        tc, tb = sub(c, test_ids), sub(b, test_ids)
        tc[name], tb[name] = lgbm_predict(booster, tc, feats), lgbm_predict(booster, tb, feats)
        preds_c.append(tc), preds_b.append(tb)
    pc, pb = pd.concat(preds_c, ignore_index=True), pd.concat(preds_b, ignore_index=True)
    return per_video(pc, pb, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,rel_rk,rel_cx,rel_both,label_peak,label_g10")
    ap.add_argument("--manifest", default="data/manifests/youtube_mr_status_v1.csv")
    ap.add_argument("--out", default="reports/iteration2_dev.csv")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging("WARNING")
    set_seeds(cfg.seed)
    st = pd.read_csv(args.manifest)
    ids = st.loc[st["ok"] == True, "video_id"].tolist()  # noqa: E712
    cand, bins = load_table(cfg, "sentence", ids), load_table(cfg, "bin", ids)
    videos = cand.groupby("video_id").agg(channel_id=("channel_id", "first")).reset_index()
    print(f"desarrollo: {len(videos)} videos, {len(cand)} candidatos")
    res = {v: cv_variant(v, cand, bins, videos, cfg) for v in args.variants.split(",")}
    rows = []
    for v, pv in res.items():
        r = {"variante": v}
        for m in ("P@5", "MRR", "NDCG@10", "tF1@5", "bin_f1@5"):
            r[m] = pv[m].mean()
            if v != "base":
                a, b = res["base"][m], pv[m]
                ok = a.notna() & b.notna()
                d = (b[ok] - a[ok]).to_numpy()
                r[f"p_{m}"] = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        rows.append(r)
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
