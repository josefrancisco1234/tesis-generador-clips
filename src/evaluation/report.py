"""Genera reports/results.md con las tablas de la validación cruzada (media ± desviación)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MAIN_METRICS = ["clip_P@1", "clip_P@5", "clip_MRR", "clip_NDCG@10", "clip_Spearman", "clip_tF1@5",
                "bin_f1@5", "bin_ap@10", "bin_hit@5"]
MODELS = ["lgbm_full", "mlp_ranknet", "opus_heuristic", "tfidf_only", "keybert_only", "audio_rms_only",
          "position_prior", "random"]
ABLATION = ["lgbm_full", "ablate_-K", "ablate_-S", "ablate_-A", "ablate_-H", "ablate_-F", "ablate_+P",
            "only_K", "only_S", "only_A", "only_H", "only_F"]
LABELS = {"lgbm_full": "LightGBM LambdaRank (propuesto)", "mlp_ranknet": "MLP RankNet (baseline neuronal)",
          "opus_heuristic": "Heurística tipo Opus (pesos fijos)", "tfidf_only": "Sólo TF-IDF KVSUM",
          "keybert_only": "Sólo KeyBERT", "audio_rms_only": "Sólo energía RMS", "position_prior": "Prior de posición",
          "random": "Aleatorio", "lgbm_label_binary": "LightGBM con etiqueta binaria",
          "ablate_-K": "− keywords (5,6)", "ablate_-S": "− sentimiento (7)", "ablate_-A": "− audio (8)",
          "ablate_-H": "− hook", "ablate_-F": "− flow", "ablate_+P": "+ posición",
          "only_K": "sólo keywords", "only_S": "sólo sentimiento", "only_A": "sólo audio", "only_H": "sólo hook",
          "only_F": "sólo flow"}


def _table(agg: pd.DataFrame, models: list[str], metrics: list[str], delta_ref: str | None = None) -> str:
    piv_m = agg.pivot(index="model", columns="metric", values="mean")
    piv_s = agg.pivot(index="model", columns="metric", values="std")
    metrics = [m for m in metrics if m in piv_m.columns]
    head = "| Modelo | " + " | ".join(metrics) + " |\n|---|" + "---|" * len(metrics) + "\n"
    rows = []
    for m in models:
        if m not in piv_m.index:
            continue
        cells = []
        for k in metrics:
            v, s = piv_m.loc[m, k], piv_s.loc[m, k]
            cell = f"{v:.3f} ± {s:.3f}"
            if delta_ref and m != delta_ref and delta_ref in piv_m.index:
                d = v - piv_m.loc[delta_ref, k]
                cell += f" ({d:+.3f})"
            cells.append(cell)
        name = LABELS.get(m, m)
        rows.append(f"| {'**' + name + '**' if m == 'lgbm_full' else name} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def build(reports: Path) -> str:
    agg = pd.read_csv(reports / "cv_metrics_long.csv")
    summary = json.loads((reports / "summary.json").read_text(encoding="utf-8"))
    parts = [
        "# Resultados (generado por run_all.py)", "",
        f"Dataset: {summary['n_videos']} videos, {summary['hours']:.1f} h, {summary['n_channels']} canales, "
        f"{summary['n_candidates']} candidatos; idiomas {summary['langs']}. Validación cruzada agrupada por canal, "
        f"k={summary['n_folds']}. Valores: media ± desviación entre folds.", "",
        "## 1. Comparación con líneas base", "", _table(agg, MODELS, MAIN_METRICS),
        "## 2. Ablación por técnica (Δ respecto al modelo completo)", "",
        _table(agg, ABLATION, ["clip_P@5", "clip_MRR", "clip_NDCG@10", "clip_tF1@5", "bin_f1@5"], "lgbm_full"),
        "## 3. Fuente de etiqueta", "",
        _table(agg, ["lgbm_full", "lgbm_label_binary"], ["clip_P@5", "clip_MRR", "clip_NDCG@10", "clip_tF1@5", "bin_f1@5"]),
        "## 4. Segmentación: oraciones vs ventanas fijas de 20 s (P4)", "",
        _table(agg, ["lgbm_full", "lgbm_trained_on_fixed20", "random"], ["clip_tF1@5", "fixed_tF1@5", "fixed_tP@5", "fixed_tR@5"]),
        "## 5. Por idioma (modelo propuesto)", "",
        _table(agg, ["lgbm_full[es]", "lgbm_full[en]"], MAIN_METRICS),
    ]
    text = "\n".join(parts)
    (reports / "results.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    from src.common.config import load_config

    print(build(load_config().path("reports")))
