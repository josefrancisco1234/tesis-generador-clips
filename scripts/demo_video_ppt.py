"""Demo en vivo: como la formula V0 y el modelo fuera-de-fold le atinan a UN video (el del PPT) y validacion con su calor.

Video: -Gzz2LZGowU (podcast de desarrollo de videojuegos indie). Uso: python scripts/demo_video_ppt.py [video_id]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scripts.reverse_engineering_formula import heat_intervals, top_k_nms  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.data.youtube_dataset import transcript_path  # noqa: E402
from src.ranking.labels import corrected_graph  # noqa: E402
from src.ranking.retention import calibrate, formula_raw, load_formula  # noqa: E402
from src.transcribe.whisper import Transcript  # noqa: E402

vid = sys.argv[1] if len(sys.argv) > 1 else "-Gzz2LZGowU"
cfg = load_config("configs/gpu.yaml")
meta = json.loads((cfg.path("interim") / "heatmaps" / f"{vid}.json").read_text(encoding="utf-8"))
df = pd.read_parquet(cfg.path("processed") / "features" / "bin" / f"{vid}.parquet")
st, en, vals = corrected_graph(meta["heatmap"])
heat_bins = heat_intervals((st, en, vals))
peak_bin = int(np.argmax(vals))
fmt = lambda s: f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"

_f = load_formula(ROOT / "models")
weights, table = _f["weights"], _f["calibrator"]
raw = formula_raw(df, weights)
score = calibrate(raw, table)
S, E = df["start"].to_numpy(), df["end"].to_numpy()
top = top_k_nms(S, E, raw, k=5)

base = np.load(ROOT / "reports" / "dl_secuencia_D054" / "oof.npz", allow_pickle=True)
vids = [str(v) for v in base["vids"]]
oof_top = top_k_nms(S, E, base["LGBM_bin"][vids.index(vid)], k=5) if vid in vids else []

tr = Transcript.load(transcript_path(cfg, vid, "base"))
ws = np.array([w.start for w in tr.words])
text_of = lambda a, b: "".join(w.text for w in tr.words[np.searchsorted(ws, a):np.searchsorted(ws, b)]).strip()
hot = lambda i: any(min(E[i], b) - max(S[i], a) > 0 for a, b in heat_bins)
heat_clip = lambda i: float(np.mean(vals[(en > S[i]) & (st < E[i])])) if ((en > S[i]) & (st < E[i])).any() else float("nan")

print(f"\nVIDEO: {meta['title']}\nCanal: {meta['channel']} | duracion {fmt(meta['duration'])} | https://youtu.be/{vid}")
print(f"Pico REAL de YouTube (Most Replayed): bin {peak_bin}  {fmt(st[peak_bin])}-{fmt(en[peak_bin])}  valor {vals[peak_bin]:.2f}")
print(f"Calor medio del video: {vals.mean():.2f} | bins calientes (>0.5): {len(heat_bins)}\n")
print("=== FORMULA V0 (17 pesos, solo contenido; NO ve la curva) -> top-5 clips ===")
for r, i in enumerate(top, 1):
    print(f" #{r} {fmt(S[i])}-{fmt(E[i])}  score {score[i]:2d}/99  calor real {heat_clip(i):.2f}  {'ACIERTA' if hot(i) else 'no toca pico'}")
    print(f"     \"{text_of(S[i], E[i])[:150]}...\"")
print(f"\nhit@5 formula: {'SI' if any(hot(i) for i in top) else 'NO'} | clip #1 sobre el pico real: "
      f"{'SI' if S[top[0]] <= en[peak_bin] and E[top[0]] >= st[peak_bin] else 'NO'}")
dist = min(abs((S[i] + E[i]) / 2 - (st[peak_bin] + en[peak_bin]) / 2) for i in top)
print(f"Distancia del clip mas cercano al pico real: {dist:.0f} s")
if oof_top:
    print(f"hit@5 LightGBM (fuera de fold): {'SI' if any(hot(i) for i in oof_top) else 'NO'}")
print(f"Azar (5 clips al azar): tocar un pico con prob ~{1 - np.prod([(100 - len(heat_bins) - k) / (100 - k) for k in range(5)]):.0%}")


# --- modo "el modelo propone, la curva valida" (el de ranking.method=auto cuando hay heatmap) ---
hv = np.array([heat_clip(i) for i in range(len(df))])
ok = hv >= np.nanpercentile(hv, 80)
fus = np.where(ok, raw, -np.inf)
ftop = top_k_nms(S, E, fus, k=5)
print("\n=== FUSION: valida con el 20 % mas caliente de la curva y ordena por la formula ===")
for r, i in enumerate(ftop, 1):
    print(f" #{r} {fmt(S[i])}-{fmt(E[i])}  score {score[i]:2d}/99  calor real {hv[i]:.2f}  {'ACIERTA' if hot(i) else 'no toca pico'}")
    print(f"     \"{text_of(S[i], E[i])[:150]}...\"")
print(f"hit@5 fusion: {'SI' if any(hot(i) for i in ftop) else 'NO'} | clip #1 sobre el pico real: "
      f"{'SI' if S[ftop[0]] <= en[peak_bin] and E[ftop[0]] >= st[peak_bin] else 'NO'}")
print(f"\nQue se dice en el pico real ({fmt(st[peak_bin])}): \"{text_of(st[peak_bin], en[peak_bin])[:260]}\"")

fig, ax = plt.subplots(figsize=(13, 4.8))
mid = (st + en) / 2
ax.fill_between(mid / 60, vals, color="#d9d9d9", step="mid")
ax.plot(mid / 60, vals, color="#555", lw=1)
ax.axhline(0.5, color="#999", ls=":", lw=1)
for k, i in enumerate(top, 1):
    ax.axvspan(S[i] / 60, E[i] / 60, color="#1b9e77", alpha=0.45)
    ax.text((S[i] + E[i]) / 120, 1.04, f"#{k}\n{score[i]}", ha="center", fontsize=9, color="#0b5d45")
for i in ftop:
    ax.axvspan(S[i] / 60, E[i] / 60, ymax=0.6, color="#7570b3", alpha=0.6)
ax.axvline(mid[peak_bin] / 60, color="#d62728", lw=2)
ax.text(mid[peak_bin] / 60, 0.02, " pico real de YouTube", color="#d62728", fontsize=9, rotation=90, va="bottom")
ax.set_ylim(0, 1.18)
ax.set_xlabel("minuto del video")
ax.set_ylabel("calor (Most Replayed)")
ax.set_title(f"{meta['title'][:70]}\nverde = formula sin ver la curva | morado = fusion (valida con la curva) | gris = calor real de YouTube")
out = ROOT / "reports" / "demo_video_ppt"
out.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(out / f"demo_{vid}.png", dpi=130)
print("\nGrafico:", out / f"demo_{vid}.png")
