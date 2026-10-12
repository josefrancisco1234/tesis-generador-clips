"""Ingenieria inversa de la "formula del exito" (EXPLORATORIO, D-044 / D-045).

Pregunta: que tienen en comun los momentos mas re-reproducidos (curva Most Replayed)
y los clips que elige el modelo / un editor humano, y se puede escribir como una
formula lineal sencilla sobre las features de contenido?

Modos:
  (por defecto)  10 videos de D-040, 1 por canal            -> reports/reverse_engineering
  --dev121       los 121 videos del conjunto de desarrollo  -> reports/reverse_engineering_dev121

Reglas del proyecto que se respetan (CLAUDE.md):
  - Solo conjunto de DESARROLLO (D-038). No se tocan los 28 videos de replica de D-040
    ni los videos de confirmacion de H-U1.
  - Unidad estadistica = VIDEO (en dev121 se promedia por CANAL antes de contrastar,
    porque hay hasta 2 videos por canal).
  - Se controla la posicion (p_rel_pos): la curva decae con el tiempo (D-018).
  - Pesos y seleccion de features se calculan dejando afuera el CANAL del video evaluado.
  - Regla de seleccion fijada ANTES de correr dev121: |rho parcial mediano| >= 0.05 y
    prueba del signo con Bonferroni (0.05 / n_features) sobre canales.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.common.config import load_config  # noqa: E402
from src.ranking.labels import corrected_graph  # noqa: E402

MODE = "dev121" if "--dev121" in sys.argv else "d040_10"
SEED = 42
N_VIDEOS = 10
TOP_K = 5
NMS_OVERLAP = 0.30           # D-019
MIN_ABS_RHO_SMALL, MIN_SIGN_AGREE = 0.08, 8      # regla para n pequeno (10 videos)
MIN_ABS_RHO_BIG = 0.05                           # regla para n grande (>=30 canales)
PREFIXES = ("k_", "s_", "a_", "h_", "f_", "e_", "l_")   # sin p_ (posicion) ni u_ (HuBERT)
OUT = ROOT / "reports" / ("reverse_engineering_dev121" if MODE == "dev121" else "reverse_engineering")


# ------------------------------------------------------------------ utilidades
def merge_iv(iv):
    iv = sorted((float(a), float(b)) for a, b in iv if b > a)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def total(iv):
    return sum(b - a for a, b in iv)


def inter(a, b):
    a, b, s, i, j = merge_iv(a), merge_iv(b), 0.0, 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        s += max(0.0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return s


def f1_temporal(clips, heat):
    if not clips or not heat:
        return float("nan")
    x = inter(clips, heat)
    p, r = x / total(merge_iv(clips)), x / total(merge_iv(heat))
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def top_k_nms(starts, ends, scores, k=TOP_K, overlap=NMS_OVERLAP):
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    keep = []
    for i in order:
        ok = True
        for j in keep:
            ov = max(0.0, min(ends[i], ends[j]) - max(starts[i], starts[j]))
            if ov / max(min(ends[i] - starts[i], ends[j] - starts[j]), 1e-9) > overlap:
                ok = False
                break
        if ok:
            keep.append(i)
        if len(keep) == k:
            break
    return keep


def ndcg_at(scores, grades, k=10):
    scores, grades = np.asarray(scores, float), np.asarray(grades, float)
    if len(scores) == 0 or np.all(grades == grades[0]):
        return float("nan")
    gain = 2.0 ** grades - 1.0
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    top = np.argsort(-scores, kind="stable")[:k]
    best = np.sort(gain)[::-1][:k]
    return float((gain[top] * disc[: len(top)]).sum() / (best * disc[: len(best)]).sum())


def p_at(scores, rel, k=TOP_K):
    top = np.argsort(-np.asarray(scores, float), kind="stable")[:k]
    return float(np.asarray(rel, float)[top].mean())


def partial_spearman(x, y, pos):
    """Spearman de x con y controlando la posicion (rango y rango^2) por residuos."""
    if np.isnan(y).any() or np.isnan(x).any():
        return float("nan")
    rx, ry, rp = (stats.rankdata(v) for v in (x, y, pos))
    rp = (rp - rp.mean()) / (rp.std() or 1.0)
    z = np.column_stack([np.ones_like(rp), rp, rp ** 2])
    res = []
    for r in (rx, ry):
        beta, *_ = np.linalg.lstsq(z, r, rcond=None)
        res.append(r - z @ beta)
    if res[0].std() < 1e-12 or res[1].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(res[0], res[1])[0, 1])


def safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.isnan(a).any() or np.isnan(b).any() or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(stats.spearmanr(a, b)[0])


# ------------------------------------------------------------------ datos
def select_videos(cfg):
    if MODE == "dev121":
        d = json.load(open(ROOT / "reports" / "conjunto_desarrollo_D038.json", encoding="utf-8"))
        return sorted(d["video_ids"])
    h = json.load(open(ROOT / "reports" / "conjunto_humano_D040.json", encoding="utf-8"))
    ids = h["d040_muestra_original"]
    rng = np.random.default_rng(SEED)
    chosen, seen = [], set()
    for vid in rng.permutation(sorted(ids)):
        meta = json.load(open(cfg.path("interim") / "heatmaps" / f"{vid}.json", encoding="utf-8"))
        if meta["channel_id"] in seen:
            continue
        seen.add(meta["channel_id"])
        chosen.append(str(vid))
        if len(chosen) == N_VIDEOS:
            break
    return chosen


def load_video(cfg, vid, cv_by_video):
    df = pd.read_parquet(cfg.path("processed") / "features" / "sentence" / f"{vid}.parquet")
    df = df.merge(cv_by_video[vid], on=["start", "end"], how="left")
    for c in ("y_human_overlap", "y_human"):
        if c not in df:
            df[c] = np.nan
    meta = json.load(open(cfg.path("interim") / "heatmaps" / f"{vid}.json", encoding="utf-8"))
    st, en, vals = corrected_graph(meta["heatmap"])
    hp = cfg.path("interim") / "human_clips" / f"{vid}.json"
    hum = []
    if hp.exists() and df["y_human_overlap"].notna().any():
        hum = [(a, b) for a, b in json.load(open(hp, encoding="utf-8"))["segments"]]
    return df, (st, en, vals), hum, meta


def heat_intervals(bins):
    st, en, vals = bins
    idx = np.nonzero(vals > 0.5)[0]
    idx = idx[np.argsort(vals[idx])[::-1]][:20]
    if len(idx) == 0:
        idx = np.argsort(vals)[::-1][:5]
    return [(st[i], en[i]) for i in idx]


# ------------------------------------------------------------------ analisis
def main():
    cfg = load_config("configs/gpu.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    cv = pd.read_parquet(ROOT / "reports" / "cv_predictions_cand.parquet")
    keepc = ["start", "end", "lgbm_autogroups", "position_prior", "random"]
    cv_by_video = {v: g[keepc] for v, g in cv.groupby("video_id")}
    vids = select_videos(cfg)
    print(f"modo={MODE}  videos={len(vids)}")

    data, feats_all = {}, None
    for v in vids:
        df, bins, hum, meta = load_video(cfg, v, cv_by_video)
        assert df["lgbm_autogroups"].notna().all(), f"merge con CV incompleto en {v}"
        feats = [c for c in df.columns if c.startswith(PREFIXES)]
        feats_all = feats if feats_all is None else [c for c in feats_all if c in feats]
        data[v] = dict(df=df, bins=bins, hum=hum, meta=meta)
    const_frac = {f: np.mean([data[v]["df"][f].nunique() <= 1 for v in vids]) for f in feats_all}
    feats = [f for f in feats_all if const_frac[f] < 0.2]
    print(f"{len(feats)} features candidatas (sin posicion, sin HuBERT)")
    chan = {v: data[v]["meta"]["channel_id"] for v in vids}
    n_chan = len(set(chan.values()))
    n_hum = sum(1 for v in vids if data[v]["hum"])
    print(f"canales={n_chan}  videos con clips humanos={n_hum}")

    for v in vids:
        d = data[v]["df"]
        data[v]["rk"] = d[feats].rank(pct=True).fillna(0.5)
        data[v]["pos"] = d["p_rel_pos"].to_numpy()

    targets = {"heat": "y_mr", "human": "y_human_overlap"}
    rho = {t: pd.DataFrame(np.nan, index=feats, columns=vids) for t in targets}
    rho_raw = {t: pd.DataFrame(np.nan, index=feats, columns=vids) for t in targets}
    for v in vids:
        d = data[v]["df"]
        for t, col in targets.items():
            y = d[col].to_numpy(dtype=float)
            if np.isnan(y).any():
                continue
            for f in feats:
                x = data[v]["rk"][f].to_numpy()
                rho[t].loc[f, v] = partial_spearman(x, y, data[v]["pos"])
                rho_raw[t].loc[f, v] = safe_spearman(x, y)

    def unit_matrix(t, train_vids):
        r = rho[t][train_vids]
        r = r.loc[:, r.notna().any(axis=0)]
        if r.shape[1] == 0:
            return r
        if MODE == "dev121":   # promedio por canal: los videos del mismo canal no son independientes
            ch = pd.Series({v: chan[v] for v in r.columns})
            return r.T.groupby(ch).mean().T
        return r

    def formula_weights(t, train_vids):
        rc = unit_matrix(t, train_vids)
        n = rc.shape[1]
        if n == 0:
            return pd.Series(dtype=float)
        med = rc.median(axis=1)
        npos, nneg = (rc > 0).sum(axis=1), (rc < 0).sum(axis=1)
        mx = np.maximum(npos, nneg)
        if MODE == "dev121" and n >= 30:
            alpha = 0.05 / len(feats)
            p = pd.Series({f: stats.binomtest(int(mx[f]), int(npos[f] + nneg[f]), 0.5).pvalue
                           for f in rc.index if npos[f] + nneg[f] > 0})
            sel = (med.abs() >= MIN_ABS_RHO_BIG) & (p.reindex(med.index) < alpha)
        elif MODE == "dev121":
            sel = (med.abs() >= MIN_ABS_RHO_SMALL) & (mx >= int(np.ceil(0.8 * n)))
        else:
            need = MIN_SIGN_AGREE if n == N_VIDEOS else MIN_SIGN_AGREE - 1
            sel = (med.abs() >= MIN_ABS_RHO_SMALL) & (mx >= need)
        return med[sel.fillna(False)]

    rows = []
    for f in feats:
        r = {"feature": f}
        for t in targets:
            vals = rho[t].loc[f].dropna()
            npos = int((vals > 0).sum())
            r[f"{t}_n_videos"] = int(len(vals))
            r[f"{t}_rho_parcial_mediano"] = float(vals.median()) if len(vals) else float("nan")
            r[f"{t}_rho_crudo_mediano"] = float(rho_raw[t].loc[f].dropna().median()) if len(vals) else float("nan")
            r[f"{t}_videos_positivos"] = npos
            r[f"{t}_p_signo"] = float(stats.binomtest(npos, len(vals), 0.5).pvalue) if len(vals) else float("nan")
        rows.append(r)
    tab = pd.DataFrame(rows).sort_values("heat_rho_parcial_mediano", key=np.abs, ascending=False)
    tab.to_csv(OUT / "features_por_video_resumen.csv", index=False, encoding="utf-8-sig")
    w_heat, w_hum = formula_weights("heat", vids), formula_weights("human", vids)

    # ---------- LOVO (dejando afuera el CANAL del video evaluado)
    per_video = []
    rng = np.random.default_rng(SEED)
    cache_w: dict = {}
    for v in vids:
        key = chan[v]
        if key not in cache_w:
            tr = [x for x in vids if chan[x] != key]
            cache_w[key] = (formula_weights("heat", tr), formula_weights("human", tr))
        wh, wu = cache_w[key]
        d, rk = data[v]["df"], data[v]["rk"]
        heat_iv, hum_iv = heat_intervals(data[v]["bins"]), data[v]["hum"]
        st, en = d["start"].to_numpy(), d["end"].to_numpy()
        heat_w = np.nan_to_num(d["y_mr"].to_numpy(), nan=0.0)
        ok_q = ((d["f_starts_dangling"] == 0) & (d["f_ends_complete"] == 1)).to_numpy()
        s_heat = (rk[wh.index].to_numpy() @ wh.to_numpy()) if len(wh) else np.zeros(len(d))
        s_hum = (rk[wu.index].to_numpy() @ wu.to_numpy()) if len(wu) else np.zeros(len(d))
        s_form_q = np.where(ok_q, s_heat - s_heat.min() + 1.0, 0.0)
        row = {"video_id": v, "channel_id": key, "dur_min": float(data[v]["meta"]["duration"] / 60),
               "n_candidatos": len(d), "n_clips_humanos": len(hum_iv)}
        mean_heat = float(d["y_mr"].mean())
        cand_scores = {"modelo_lgbm": d["lgbm_autogroups"].to_numpy(),
                       "formula_heat": s_heat, "formula_human": s_hum,
                       "formula_mas_calidad": s_form_q,
                       "posicion": d["position_prior"].to_numpy(),
                       "calor_directo": heat_w,
                       "calor_mas_calidad": np.where(ok_q, heat_w, -1.0)}
        for name, sc in cand_scores.items():
            keep = top_k_nms(st, en, sc)
            clips = [(st[i], en[i]) for i in keep]
            row[f"{name}_pct_inicio_colgado"] = float(d["f_starts_dangling"].to_numpy()[keep].mean())
            row[f"{name}_pct_termina_completo"] = float(d["f_ends_complete"].to_numpy()[keep].mean())
            row[f"{name}_tF1_vs_calor"] = f1_temporal(clips, heat_iv)
            row[f"{name}_tF1_vs_humano"] = f1_temporal(clips, hum_iv) if hum_iv else float("nan")
            row[f"{name}_lift_calor"] = float(d["y_mr"].to_numpy()[keep].mean() / mean_heat)
            row[f"{name}_spearman_calor"] = safe_spearman(sc, d["y_mr"])
            row[f"{name}_spearman_humano"] = safe_spearman(sc, d["y_human_overlap"]) if hum_iv else float("nan")
            row[f"{name}_ndcg10_calor"] = ndcg_at(sc, d["y_grade"])
            row[f"{name}_P5_calor"] = p_at(sc, d["y_relevant"])
            row[f"{name}_P5_humano"] = p_at(sc, d["y_human"]) if hum_iv else float("nan")
        row["humano_tF1_vs_calor"] = f1_temporal(hum_iv, heat_iv) if hum_iv else float("nan")
        inside = [d["y_mr"].to_numpy()[(d["start"] >= a - 1) & (d["end"] <= b + 1)] for a, b in hum_iv]
        inside = np.concatenate([x for x in inside if len(x)]) if any(len(x) for x in inside) else np.array([])
        row["humano_lift_calor"] = float(inside.mean() / mean_heat) if len(inside) else float("nan")
        draws = []
        for _ in range(100):
            keep = top_k_nms(st, en, rng.random(len(d)))
            cl = [(st[i], en[i]) for i in keep]
            draws.append((f1_temporal(cl, heat_iv), f1_temporal(cl, hum_iv) if hum_iv else np.nan,
                          float(d["y_mr"].to_numpy()[keep].mean() / mean_heat)))
        row["azar_tF1_vs_calor"], row["azar_tF1_vs_humano"], row["azar_lift_calor"] = np.nanmean(draws, axis=0)
        row["n_feats_formula_heat"], row["n_feats_formula_human"] = int(len(wh)), int(len(wu))
        row["rho_calor_vs_humano"] = safe_spearman(d["y_mr"], d["y_human_overlap"]) if hum_iv else float("nan")
        per_video.append(row)
    pv = pd.DataFrame(per_video)
    pv.to_csv(OUT / "por_video_LOVO.csv", index=False, encoding="utf-8-sig")

    def mean_ci(col):
        x = pv[col].dropna().to_numpy()
        return (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))) if len(x) > 1 else (float("nan"),) * 2

    methods = ["azar", "posicion", "modelo_lgbm", "formula_heat", "formula_mas_calidad", "formula_human",
               "calor_directo", "calor_mas_calidad", "humano"]
    metrics = ["tF1_vs_calor", "tF1_vs_humano", "lift_calor", "spearman_calor", "spearman_humano",
               "ndcg10_calor", "P5_calor", "P5_humano", "pct_inicio_colgado", "pct_termina_completo"]
    summary = {"modo": MODE, "n_videos": len(vids), "n_canales": n_chan, "n_videos_con_humano": n_hum,
               "videos": vids, "n_features_candidatas": len(feats),
               "formula_calor": {k: float(x) for k, x in w_heat.sort_values(key=np.abs, ascending=False).items()},
               "formula_humano": {k: float(x) for k, x in w_hum.sort_values(key=np.abs, ascending=False).items()},
               "metricas": {}}
    for m in methods:
        summary["metricas"][m] = {}
        for k in metrics:
            col = f"{m}_{k}"
            if col in pv and pv[col].notna().any():
                mu, se = mean_ci(col)
                summary["metricas"][m][k] = {"media": mu, "ee": se, "n": int(pv[col].notna().sum())}
    pairs = [("formula_heat", "azar", "tF1_vs_calor"), ("modelo_lgbm", "azar", "tF1_vs_calor"),
             ("formula_heat", "modelo_lgbm", "tF1_vs_calor"), ("formula_heat", "modelo_lgbm", "ndcg10_calor"),
             ("formula_heat", "posicion", "spearman_calor"), ("formula_heat", "azar", "lift_calor"),
             ("formula_mas_calidad", "formula_heat", "pct_inicio_colgado"),
             ("formula_mas_calidad", "azar", "tF1_vs_calor"),
             ("formula_human", "azar", "tF1_vs_humano"), ("modelo_lgbm", "azar", "tF1_vs_humano"),
             ("humano", "azar", "tF1_vs_calor"), ("calor_directo", "modelo_lgbm", "tF1_vs_calor"),
             ("calor_directo", "azar", "tF1_vs_humano"),
             ("calor_mas_calidad", "calor_directo", "lift_calor"),
             ("calor_mas_calidad", "modelo_lgbm", "tF1_vs_calor"),
             ("calor_mas_calidad", "calor_directo", "pct_inicio_colgado")]
    tests = {}
    for a, b, k in pairs:
        ca, cb = f"{a}_{k}", f"{b}_{k}"
        if ca in pv and cb in pv:
            x, y = pv[ca].to_numpy(float), pv[cb].to_numpy(float)
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 5:
                continue
            try:
                p = float(stats.wilcoxon(x[ok], y[ok], alternative="two-sided").pvalue)
            except ValueError:
                p = float("nan")
            tests[f"{a} vs {b} [{k}]"] = {"n": int(ok.sum()), "dif_media": float((x[ok] - y[ok]).mean()),
                                          "mejor_en": int((x[ok] > y[ok]).sum()), "p_wilcoxon": p}
    summary["contrastes"] = tests
    rc = pv.rho_calor_vs_humano.dropna()
    summary["rho_calor_vs_humano"] = {"n": int(len(rc)), "media": float(rc.mean()) if len(rc) else None,
                                      "mediana": float(rc.median()) if len(rc) else None}
    both = sorted(set(w_heat.index) | set(w_hum.index))
    if both:
        a, b = w_heat.reindex(both).fillna(0), w_hum.reindex(both).fillna(0)
        summary["concordancia_formulas"] = {
            "n_features_union": len(both),
            "mismo_signo": int(((np.sign(a) == np.sign(b)) & (a != 0) & (b != 0)).sum()),
            "signo_opuesto": int(((np.sign(a) * np.sign(b)) < 0).sum()),
            "solo_calor": int(((a != 0) & (b == 0)).sum()), "solo_humano": int(((a == 0) & (b != 0)).sum())}
    json.dump(summary, open(OUT / "resumen.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    make_figures(vids, data, pv, tab, w_heat)
    print_report(summary, pv)


# ------------------------------------------------------------------ figuras
def make_figures(vids, data, pv, tab, w_heat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [v for v in vids if data[v]["hum"]][:10] or vids[:10]
    fig, axes = plt.subplots(5, 2, figsize=(15, 15))
    for ax, v in zip(axes.ravel(), show):
        d, (st, en, vals), hum, meta = data[v]["df"], data[v]["bins"], data[v]["hum"], data[v]["meta"]
        mid = (st + en) / 2 / 60
        ax.fill_between(mid, vals, color="#d9534f", alpha=0.18)
        ax.plot(mid, vals, color="#d9534f", lw=1.3)
        s_, e_ = d["start"].to_numpy(), d["end"].to_numpy()
        for i in top_k_nms(s_, e_, d["lgbm_autogroups"].to_numpy()):
            ax.axvspan(s_[i] / 60, e_[i] / 60, ymin=0.0, ymax=0.88, color="#2b6cb0", alpha=0.28)
        for a, b in hum:
            ax.hlines(-0.06, a / 60, b / 60, colors="#2f855a", lw=6)
        row = pv[pv.video_id == v].iloc[0]
        ax.set_title(f"{v} | {meta['duration']/60:.0f} min | F1 modelo-calor {row.modelo_lgbm_tF1_vs_calor:.2f}",
                     fontsize=9)
        ax.set_ylim(-0.12, 1.05)
    handles = [plt.Line2D([0], [0], color="#d9534f", lw=2), plt.Rectangle((0, 0), 1, 1, color="#2b6cb0", alpha=0.3),
               plt.Line2D([0], [0], color="#2f855a", lw=6)]
    fig.legend(handles, ["curva Most Replayed", "top-5 del modelo (fuera de fold)", "clips de editor humano"],
               loc="upper center", ncol=3, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig1_calor_vs_clips_10_videos.png", dpi=110)
    plt.close(fig)

    top = tab.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(top))
    ax.barh(y + 0.2, top["heat_rho_parcial_mediano"], 0.4, color="#d9534f", label="vs calor (y_mr)")
    ax.barh(y - 0.2, top["human_rho_parcial_mediano"].fillna(0), 0.4, color="#2f855a", label="vs clips humanos")
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("rho parcial mediano (controlando posicion)")
    ax.set_title(f"Features que acompanan al calor (top 20 por |rho|), n={len(vids)} videos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_features_calor_vs_humano.png", dpi=110)
    plt.close(fig)

    meths = [("azar", "azar"), ("posicion", "solo posicion"), ("modelo_lgbm", "modelo actual"),
             ("formula_heat", "formula contenido"), ("formula_mas_calidad", "formula + calidad"),
             ("calor_mas_calidad", "calor + calidad"), ("calor_directo", "calor directo")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (suf, ttl) in zip(axes, [("tF1_vs_calor", "F1 temporal de los top-5 vs picos"),
                                     ("lift_calor", "calor medio de los clips / calor medio del video")]):
        arr = [pv[f"{m}_{suf}"].dropna().to_numpy() for m, _ in meths]
        ax.boxplot(arr, tick_labels=[t for _, t in meths], showmeans=True)
        ax.set_title(ttl)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        if suf == "lift_calor":
            ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_metodos_comparados.png", dpi=110)
    plt.close(fig)


def print_report(s, pv):
    print(f"\n=== FORMULA (calor, n={s['n_videos']} videos / {s['n_canales']} canales) ===")
    for k, v in s["formula_calor"].items():
        print(f"  {v:+.3f} * rk({k})")
    print(f"\n=== FORMULA (humano, {s['n_videos_con_humano']} videos) ===")
    for k, v in s["formula_humano"].items():
        print(f"  {v:+.3f} * rk({k})")
    print("\n=== METRICAS (media +- ee) ===")
    for m, d in s["metricas"].items():
        print(m.ljust(20), {k: f"{x['media']:.3f}+-{x['ee']:.3f}" for k, x in d.items()
                            if k in ("tF1_vs_calor", "tF1_vs_humano", "lift_calor", "spearman_calor",
                                     "ndcg10_calor", "pct_inicio_colgado")})
    print("\n=== CONTRASTES (Wilcoxon bilateral, exploratorio) ===")
    for k, t in s["contrastes"].items():
        print(f"  {k}: n={t['n']} dif={t['dif_media']:+.3f} mejor_en={t['mejor_en']}/{t['n']} p={t['p_wilcoxon']:.4f}")
    print("\nrho calor~humano:", s["rho_calor_vs_humano"])
    print("concordancia formulas:", s.get("concordancia_formulas"))


if __name__ == "__main__":
    main()
