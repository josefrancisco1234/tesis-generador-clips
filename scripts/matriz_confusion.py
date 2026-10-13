"""Matriz de confusion a nivel de tramo (bin) y a nivel de video, sobre los videos NO usados para fijar la formula (V1, solo contenido)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scripts.reverse_engineering_formula import heat_intervals, top_k_nms
from src.common.config import load_config
from src.ranking.labels import corrected_graph
from src.ranking.retention import formula_raw, load_formula
cfg = load_config("configs/gpu.yaml")
W = load_formula(ROOT / "models")["weights"]
dev = set(json.loads((ROOT / "reports" / "conjunto_desarrollo_D038.json").read_text(encoding="utf-8"))["video_ids"])
ids = sorted(p.stem for p in (cfg.path("processed") / "features" / "bin").glob("*.parquet") if p.stem not in dev)
TP = FP = FN = TN = 0
vids_hit = vids_miss = 0
nvid = 0
for v in ids:
    hp = cfg.path("interim") / "heatmaps" / f"{v}.json"
    if not hp.exists(): continue
    m = json.loads(hp.read_text(encoding="utf-8")); df = pd.read_parquet(cfg.path("processed") / "features" / "bin" / f"{v}.parquet")
    st, en, vals = corrected_graph(m["heatmap"]); S, E = df.start.to_numpy(), df.end.to_numpy()
    raw = formula_raw(df, W)
    sel = top_k_nms(S, E, np.where((E - S) <= 45, raw, raw.min() - 10), k=5)
    hot_idx = np.nonzero(vals > 0.5)[0]; hot_idx = hot_idx[np.argsort(vals[hot_idx])[::-1]][:20]
    if len(hot_idx) == 0: hot_idx = np.argsort(vals)[::-1][:5]
    hot = np.zeros(len(df), bool)
    for i in range(len(df)):                      # una ventana es "caliente" si se solapa con algun bin caliente
        hot[i] = any(min(E[i], en[b]) - max(S[i], st[b]) > 0 for b in hot_idx)
    pred = np.zeros(len(df), bool); pred[sel] = True
    TP += int((pred & hot).sum()); FP += int((pred & ~hot).sum()); FN += int((~pred & hot).sum()); TN += int((~pred & ~hot).sum())
    nvid += 1
    if (pred & hot).any(): vids_hit += 1
    else: vids_miss += 1
tot = TP + FP + FN + TN
prec = TP / (TP + FP); rec = TP / (TP + FN)
print(f"videos: {nvid}\nTP={TP} FP={FP} FN={FN} TN={TN}\nprecision={prec:.1%}  recall={rec:.1%}  (azar de precision = tasa base de ventanas calientes = {(TP+FN)/tot:.1%})")
print(f"videos con >=1 acierto: {vids_hit}/{nvid} = {vids_hit/nvid:.1%}")
fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
M = np.array([[TP, FP], [FN, TN]])   # filas = prediccion, columnas = realidad
ax.imshow(np.array([[1, 0], [0, 0]]), cmap="Oranges", vmin=0, vmax=2)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center", fontsize=15, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Caliente\n(pico real)", "No caliente"], fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Elegido\n(top 5)", "No elegido"], fontsize=9)
ax.set_xlabel("Realidad (curva de YouTube)", fontsize=9); ax.set_ylabel("Predicción del sistema", fontsize=9)
ax.set_title(f"Matriz de confusión por ventana, {nvid} videos\nprecisión {prec:.0%} · cobertura {rec:.0%}", fontsize=10)
fig.tight_layout(); out = ROOT / "reports" / "demo_video_ppt" / "matriz_confusion.png"; fig.savefig(out); print(out)
