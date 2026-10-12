"""Entrenamiento, evaluación, líneas base y ablaciones (Fase 3).

Protocolo:
  - Validación cruzada agrupada por canal (GroupKFold, k=5): en cada fold, el 15 % de los canales
    de entrenamiento se reserva para early stopping y calibración. Se reporta media ± desviación.
  - Modelo final: split fijo 70/15/15 por canal (semilla fija), guardado en models/.
  - Evaluación en tres granularidades: candidatos por oración (clips reales), bins de 1 % (protocolo
    Rhapsody) y ventanas fijas de 20 s (línea base Minimal Clips).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.common.config import Config, load_config, set_seeds
from src.common.logging_utils import get_logger, setup_logging
from src.data.featurize import load_table
from src.evaluation import baselines as B
from src.evaluation.metrics import bin_protocol, group_ranking_metrics, temporal_f1_by_video
from src.features.builder import GROUP_PREFIX, feature_columns
from src.ranking.model import MLPRanker, make_bundle, train_lgbm

log = get_logger(__name__)
GROUP_NAMES = {"K": "keywords (TF-IDF+KeyBERT)", "S": "sentimiento/léxico", "A": "audio (librosa)",
               "H": "hook", "F": "flow", "P": "posición"}


# ------------------------------------------------------------------ utilidades
def split_videos(videos: pd.DataFrame, cfg: Config, seed: int) -> dict[str, list[str]]:
    sp = cfg.ranking["split"]
    g = videos["channel_id"].fillna(videos["video_id"])
    gss = GroupShuffleSplit(n_splits=1, test_size=sp["test"], random_state=seed)
    trval_idx, test_idx = next(gss.split(videos, groups=g))
    trval = videos.iloc[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=sp["val"] / (sp["train"] + sp["val"]), random_state=seed)
    tr_idx, va_idx = next(gss2.split(trval, groups=g.iloc[trval_idx]))
    return {"train": trval.iloc[tr_idx]["video_id"].tolist(), "val": trval.iloc[va_idx]["video_id"].tolist(),
            "test": videos.iloc[test_idx]["video_id"].tolist()}


def inner_val(videos: pd.DataFrame, train_ids: list[str], seed: int, frac: float = 0.15) -> tuple[list[str], list[str]]:
    v = videos[videos["video_id"].isin(train_ids)]
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    a, b = next(gss.split(v, groups=v["channel_id"].fillna(v["video_id"])))
    return v.iloc[a]["video_id"].tolist(), v.iloc[b]["video_id"].tolist()


def sub(df: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    return df[df["video_id"].isin(ids)].reset_index(drop=True)


def evaluate_scores(cand: pd.DataFrame, bins: pd.DataFrame, fixed: pd.DataFrame, col: str, cfg: Config) -> dict:
    lc, k = cfg.labels, 5
    out = {}
    if col in cand:
        out.update({f"clip_{m}": v for m, v in group_ranking_metrics(cand, col).items() if m != "n_videos"})
        out.update({f"clip_{m}@{k}": v for m, v in temporal_f1_by_video(cand, col, bins, k, cfg.ranking["nms_iou"],
                                                                         lc["max_n_gt"], lc["gt_threshold"]).items()
                    if m != "n_videos"})
    if col in bins:
        for kk in (5, 10):
            out.update({f"bin_{m}@{kk}": v for m, v in bin_protocol(bins, col, kk, lc["max_n_gt"], lc["gt_threshold"]).items()
                        if m != "n_videos"})
    if col in fixed:
        out.update({f"fixed_{m}@{k}": v for m, v in temporal_f1_by_video(fixed, col, bins, k, cfg.ranking["nms_iou"],
                                                                          lc["max_n_gt"], lc["gt_threshold"]).items()
                    if m != "n_videos"})
    return out


def lgbm_predict(booster, df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return booster.predict(df[feats], num_iteration=booster.best_iteration or None)


# ------------------------------------------------------------------ un fold completo
def run_fold(tables: dict[str, pd.DataFrame], videos: pd.DataFrame, train_ids: list[str], val_ids: list[str],
             test_ids: list[str], cfg: Config, seed: int, ablations: bool = True,
             force_groups: list[str] | None = None) -> tuple[dict, dict]:
    cand, bins, fixed = tables["sentence"], tables["bin"], tables["fixed"]
    tr, va = sub(cand, train_ids), sub(cand, val_ids)
    te = {"cand": sub(cand, test_ids), "bins": sub(bins, test_ids), "fixed": sub(fixed, test_ids)}
    groups_all = cfg.features["groups"]
    # D-038: el procedimiento de D-027 y `lgbm_full` se calculan SIEMPRE sin el grupo U, para que sigan
    # siendo comparables con los resultados previos; U se evalúa en variantes paralelas (`*_U`).
    groups = [g for g in groups_all if g != "U"]
    feats = feature_columns(cand, groups)
    has_u = "U" in groups_all and bool(feature_columns(cand, ["U"]))
    feats_u = feature_columns(cand, groups_all) if has_u else None
    params = cfg.ranking["lgbm"]
    results: dict[str, dict] = {}
    models: dict = {}

    def score_all(name: str, fn) -> None:
        for key in te:
            te[key][name] = fn(te[key])
        results[name] = evaluate_scores(te["cand"], te["bins"], te["fixed"], name, cfg)

    # --- modelo principal
    booster = train_lgbm(tr, va, feats, params, seed)
    models["lgbm"] = (booster, feats, va)
    score_all("lgbm_full", lambda d: lgbm_predict(booster, d, feats))

    # --- selección automática de grupos DENTRO del fold (D-027): evita elegir el subconjunto
    #     mirando el test. Se entrena cada candidato en train y se elige por NDCG@10 en validación.
    subsets = [force_groups] if force_groups else [["A"], ["A", "H"], ["A", "E"], ["A", "L"], ["K", "A"],
                                                   ["K", "S", "A", "H", "F"], groups]
    scored: list[tuple[float, object, list[str], str]] = []

    def try_subset(sset: list[str]) -> None:
        fs = feature_columns(cand, sset)
        b = train_lgbm(tr, va, fs, params, seed)
        score = float(b.best_score.get("val", {}).get(f"ndcg@{params['eval_at'][-1]}", -1.0))
        log.info("  auto-grupos %s → NDCG val %.4f", "+".join(sset), score)
        scored.append((score, b, fs, "+".join(sset)))

    def pick(cands: list) -> tuple:
        best = max(cands, key=lambda c: c[0])  # empate → el primero, como el bucle original
        return best[1], best[2], best[3]

    for sset in subsets:
        try_subset(sset)
    best, best_feats, best_name = pick(scored)
    log.info("  auto-grupos → elegido %s", best_name)
    results["_auto_groups_elegidos"] = {"n_features": len(best_feats)}
    models["auto"] = (best, best_feats, va, best_name)
    score_all("lgbm_autogroups", lambda d: lgbm_predict(best, d, best_feats))

    # --- D-038: mismo procedimiento con 4 subconjuntos adicionales que incluyen U (fijados a priori).
    #     Los 7 subconjuntos base NO se reentrenan: se reutilizan sus puntuaciones de validación.
    if has_u and not force_groups:
        for sset in (["U"], ["A", "U"], ["A", "H", "U"], groups_all):
            try_subset(sset)
        bu, fu, nu = pick(scored)
        log.info("  auto-grupos+U → elegido %s", nu)
        models["auto_U"] = (bu, fu, va, nu)
        score_all("lgbm_autogroups_U", lambda d: lgbm_predict(bu, d, fu))
        full_u = train_lgbm(tr, va, feats_u, params, seed)
        score_all("lgbm_full_U", lambda d: lgbm_predict(full_u, d, feats_u))

    # --- baseline neuronal
    m = cfg.ranking["mlp"]
    mlp = MLPRanker(feats, m["hidden"], m["dropout"], m["lr"], m["epochs"], m["pairs_per_group"], seed).fit(tr)
    score_all("mlp_ranknet", mlp.predict)

    # --- líneas base
    score_all("random", lambda d: B.random_scores(d, seed))
    score_all("position_prior", lambda d: B.position_prior(tr, d))
    score_all("tfidf_only", lambda d: B.single_feature(d, "k_kvsum_local"))
    score_all("keybert_only", lambda d: B.single_feature(d, "k_keybert_sim"))
    score_all("audio_rms_only", lambda d: B.single_feature(d, "a_rms_mean"))
    score_all("opus_heuristic", B.opus_heuristic)

    # --- segmentación: modelo entrenado sobre ventanas fijas de 20 s (P4) evaluado en su propio dominio
    trf, vaf = sub(fixed, train_ids), sub(fixed, val_ids)
    bf = train_lgbm(trf, vaf, feats, params, seed)
    te["fixed"]["lgbm_fixed20"] = lgbm_predict(bf, te["fixed"], feats)
    results["lgbm_trained_on_fixed20"] = {
        f"fixed_{k}@5": v for k, v in temporal_f1_by_video(te["fixed"], "lgbm_fixed20", te["bins"], 5,
                                                          cfg.ranking["nms_iou"]).items() if k != "n_videos"}

    # --- etiqueta: binaria vs graduada
    bb = train_lgbm(tr.assign(y_bin=tr["y_relevant"]), va.assign(y_bin=va["y_relevant"]), feats, params, seed, "y_bin")
    score_all("lgbm_label_binary", lambda d: lgbm_predict(bb, d, feats))

    if ablations:
        for g in groups:  # quitar un grupo
            f2 = [f for f in feats if not f.startswith(GROUP_PREFIX[g])]
            b2 = train_lgbm(tr, va, f2, params, seed)
            score_all(f"ablate_-{g}", lambda d, b2=b2, f2=f2: lgbm_predict(b2, d, f2))
        for g in groups:  # sólo un grupo
            f3 = [f for f in feats if f.startswith(GROUP_PREFIX[g])]
            b3 = train_lgbm(tr, va, f3, params, seed)
            score_all(f"only_{g}", lambda d, b3=b3, f3=f3: lgbm_predict(b3, d, f3))
        if has_u:
            fu_only = [f for f in feats_u if f.startswith(GROUP_PREFIX["U"])]
            bu_only = train_lgbm(tr, va, fu_only, params, seed)
            score_all("only_U", lambda d: lgbm_predict(bu_only, d, fu_only))
        f4 = feats + feature_columns(cand, ["P"])  # añadir posición
        b4 = train_lgbm(tr, va, f4, params, seed)
        score_all("ablate_+P", lambda d: lgbm_predict(b4, d, f4))

    # --- por idioma (modelo principal)
    for lang in ("es", "en"):
        ids = [v for v in test_ids if (videos.set_index("video_id").loc[v, "lang"] or "").startswith(lang)]
        if ids:
            results[f"lgbm_full[{lang}]"] = evaluate_scores(sub(te["cand"], ids), sub(te["bins"], ids),
                                                            sub(te["fixed"], ids), "lgbm_full", cfg)
    models["test_frames"] = te
    return results, models


def summarize(fold_results: list[dict]) -> pd.DataFrame:
    rows = []
    for fi, res in enumerate(fold_results):
        for model, metrics in res.items():
            for k, v in metrics.items():
                rows.append({"fold": fi, "model": model, "metric": k, "value": v})
    long = pd.DataFrame(rows)
    agg = long.groupby(["model", "metric"])["value"].agg(["mean", "std"]).reset_index()
    return agg


# ------------------------------------------------------------------ principal
def run(cfg: Config, out_dir: Path, n_folds: int = 5, ablations: bool = True) -> dict:
    t0 = time.time()
    set_seeds(cfg.seed)
    tables = {k: load_table(cfg, k) for k in ("sentence", "bin", "fixed")}
    cand = tables["sentence"]
    if cand.empty:
        raise RuntimeError("no hay features; ejecuta primero la featurización")
    videos = (cand.groupby("video_id").agg(channel_id=("channel_id", "first"), lang=("lang", "first"),
                                           duration=("duration_video", "first"), n=("win_id", "size")).reset_index())
    log.info("dataset: %d videos, %d candidatos, %d canales", len(videos), len(cand), videos["channel_id"].nunique())
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- validación cruzada agrupada por canal
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    chosen_per_fold: list[str] = []
    chosen_u_per_fold: list[str] = []
    preds: dict[str, list[pd.DataFrame]] = {}
    for fi, (a, b) in enumerate(gkf.split(videos, groups=videos["channel_id"].fillna(videos["video_id"]))):
        trainval, test_ids = videos.iloc[a]["video_id"].tolist(), videos.iloc[b]["video_id"].tolist()
        tr_ids, va_ids = inner_val(videos, trainval, cfg.seed + fi)
        log.info("fold %d: %d train, %d val, %d test", fi, len(tr_ids), len(va_ids), len(test_ids))
        res, mods = run_fold(tables, videos, tr_ids, va_ids, test_ids, cfg, cfg.seed, ablations)
        fold_results.append(res)
        chosen_per_fold.append(mods["auto"][3])
        if "auto_U" in mods:
            chosen_u_per_fold.append(mods["auto_U"][3])
        for key in ("cand", "bins"):  # predicciones de test para pruebas de significancia por video
            preds.setdefault(key, []).append(mods["test_frames"][key].assign(fold=fi))
    agg = summarize(fold_results)
    agg.to_csv(out_dir / "cv_metrics_long.csv", index=False)
    keep = ["video_id", "fold", "start", "end", "lang", "y_mr", "y_grade", "y_relevant"]
    for key, parts in preds.items():
        df = pd.concat(parts, ignore_index=True)
        cols = keep + [c for c in df.columns if c not in df.columns[:0] and c not in keep and not c[:2] in
                       ("k_", "s_", "a_", "h_", "f_", "p_") and c not in ("win_id", "s0", "s1", "kind", "channel_id",
                                                                          "query_lang", "duration_video")]
        df[cols].to_parquet(out_dir / f"cv_predictions_{key}.parquet", index=False)

    # ---- modelo final con split fijo
    split = split_videos(videos, cfg, cfg.seed)
    (out_dir / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    majority = Counter(chosen_per_fold).most_common(1)[0][0] if chosen_per_fold else None
    log.info("grupos elegidos por fold: %s → mayoría: %s", chosen_per_fold, majority)
    res_final, models = run_fold(tables, videos, split["train"], split["val"], split["test"], cfg, cfg.seed,
                                 ablations=False, force_groups=majority.split("+") if majority else None)
    full_booster, full_feats, va = models["lgbm"]
    booster, feats, va_auto, chosen = models["auto"]
    import lightgbm as lgb
    import sklearn

    log.info("modelo final: selección automática → grupos %s (%d features)", chosen, len(feats))
    bundle = make_bundle(booster, va, feats, {
        "modelo": "lgbm_autogroups", "grupos_elegidos": chosen,
        "eleccion_por_fold": chosen_per_fold,
        "trained_on": "YT-MR (dataset propio)", "n_train_videos": len(split["train"]), "seed": cfg.seed,
        "lightgbm": lgb.__version__, "sklearn": sklearn.__version__, "python": platform.python_version(),
        "groups": cfg.features["groups"], "created": time.strftime("%Y-%m-%d %H:%M"),
    }, explainer=full_booster, explainer_features=full_feats)
    bundle.save(cfg.path("models"))
    imp = pd.DataFrame({"feature": feats, "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)
    pd.DataFrame({"feature": full_feats, "gain": full_booster.feature_importance("gain")}) \
        .sort_values("gain", ascending=False).to_csv(out_dir / "feature_importance_todos_los_grupos.csv", index=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)
    pd.DataFrame(res_final).T.to_csv(out_dir / "final_split_metrics.csv")
    _plots(imp, models["test_frames"], booster, feats, out_dir)
    if chosen_u_per_fold:  # D-038: ¿entra U en la selección automática?
        (out_dir / "grupos_elegidos_por_fold.json").write_text(json.dumps(
            {"base_D027": chosen_per_fold, "con_U_D038": chosen_u_per_fold}, indent=1), encoding="utf-8")
    summary = {"n_videos": len(videos), "n_candidates": len(cand), "n_channels": int(videos["channel_id"].nunique()),
               "langs": videos["lang"].value_counts().to_dict(), "n_folds": n_folds,
               "hours": float(videos["duration"].sum() / 3600), "elapsed_s": time.time() - t0}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log.info("experimentos terminados en %.0fs → %s", time.time() - t0, out_dir)
    return summary


def _plots(imp: pd.DataFrame, te: dict, booster, feats: list[str], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["gain"], color="#3b6ea5")
    ax.set_xlabel("ganancia total (LightGBM)")
    ax.set_title("Top-20 features por importancia")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    # curva real vs predicha en un video de test (bins)
    bins = te["bins"]
    if not bins.empty and "lgbm_full" in bins:
        vid = bins["video_id"].iloc[0]
        b = bins[bins["video_id"] == vid].sort_values("start")
        pred = (b["lgbm_full"] - b["lgbm_full"].min()) / (np.ptp(b["lgbm_full"]) + 1e-9)
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(b["start"] / 60, b["y_mr"], label="Most Replayed (real)", color="#222")
        ax.plot(b["start"] / 60, pred, label="score del modelo (normalizado)", color="#d9822b")
        ax.set_xlabel("minuto")
        ax.legend(loc="upper right")
        ax.set_title(f"Video de test {vid}")
        fig.tight_layout()
        fig.savefig(out_dir / "curve_example.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-ablations", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.logging["level"], cfg.logging["file"])
    run(cfg, cfg.path("reports"), args.folds, not args.no_ablations)


if __name__ == "__main__":
    main()
