"""Validacion SOLO CONTENIDO (como en un video no subido a YouTube): V1 elige 5 clips sin ver la curva y la curva real
se usa unicamente DESPUES, para juzgar. Cohorte fresca 1 (93 videos); se grafican los primeros 12 por orden de id (sin escoger)."""
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
ids = json.loads((ROOT / "reports" / "conjunto_confirmacion_fresca_D050.json").read_text(encoding="utf-8"))["video_ids"]
f = load_formula(ROOT / "models"); W = f["weights"]
rows = []
for v in ids:
    hp = cfg.path("interim") / "heatmaps" / f"{v}.json"; bp = cfg.path("processed") / "features" / "bin" / f"{v}.parquet"
    if not (hp.exists() and bp.exists()): continue
    m = json.loads(hp.read_text(encoding="utf-8")); df = pd.read_parquet(bp)
    st, en, vals = corrected_graph(m["heatmap"]); S, E = df.start.to_numpy(), df.end.to_numpy()
    hot = heat_intervals((st, en, vals))
    raw = formula_raw(df, W)
    top = top_k_nms(S, E, raw, k=5)
    touch = [any(min(E[i], b) - max(S[i], a) > 0 for a, b in hot) for i in top]
    rows.append(dict(v=v, title=m["title"], st=st, en=en, vals=vals, S=S, E=E, top=top, touch=touch, hit=any(touch)))
n = len(rows); h = sum(r["hit"] for r in rows)
print(f"videos: {n} | hit@5 con SOLO contenido (V0, clips elegidos sin ver la curva): {h}/{n} = {h/n:.1%}")
sel = rows[:12]
fig, axs = plt.subplots(4, 3, figsize=(15, 9))
for a, r in zip(axs.ravel(), sel):
    mid = (r["st"] + r["en"]) / 2
    a.fill_between(mid / 60, r["vals"], color="#d9d9d9", step="mid")
    for i, t in zip(r["top"], r["touch"]):
        a.axvspan(r["S"][i] / 60, r["E"][i] / 60, color="#1b9e77" if t else "#999", alpha=0.75)
    a.set_title(("ACIERTA  " if r["hit"] else "falla  ") + r["title"][:42], fontsize=8, color="#0b5d45" if r["hit"] else "#a33")
    a.set_yticks([]); a.tick_params(labelsize=7)
fig.suptitle(f"Solo contenido (sin ver la curva): verde = clip que toca un pico real, gris = clip fuera. Aciertos en los 12 mostrados: "
             f"{sum(r['hit'] for r in sel)}/12 | en TODA la cohorte: {h}/{n} = {h/n:.0%}", fontsize=11)
plt.tight_layout(); out = ROOT / "reports" / "demo_video_ppt" / "validacion_solo_contenido.png"; plt.savefig(out, dpi=120); print(out)
