"""Congela la formula de D-045 en models/formula_calor_D045.json (pesos + calibrador).

El calibrador es la distribucion del score crudo de los top-10 clips (NMS 0.3) de cada video
del conjunto de DESARROLLO (D-038), igual que el calibrador del modelo LightGBM (D-019).
No toca los videos de confirmacion.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.config import load_config  # noqa: E402
from src.ranking.retention import FORMULA_FILE, formula_raw  # noqa: E402
from scripts.reverse_engineering_formula import top_k_nms  # noqa: E402

cfg = load_config("configs/gpu.yaml")
res = json.loads((ROOT / "reports" / "reverse_engineering_dev121" / "resumen.json").read_text(encoding="utf-8"))
weights = res["formula_calor"]
dev = json.loads((ROOT / "reports" / "conjunto_desarrollo_D038.json").read_text(encoding="utf-8"))["video_ids"]
vals = []
for v in dev:
    d = pd.read_parquet(cfg.path("processed") / "features" / "sentence" / f"{v}.parquet")
    raw = formula_raw(d, weights)
    keep = top_k_nms(d["start"].to_numpy(), d["end"].to_numpy(), raw, k=10)
    vals += [float(raw[i]) for i in keep]
table = np.quantile(np.array(vals), np.arange(100) / 100).tolist()
out = {"nombre": "formula_calor_D045", "creada": "2026-10-12", "decisiones": ["D-045", "D-046", "D-047"],
       "descripcion": "Suma ponderada de rangos percentiles (dentro del video) de features de contenido; "
                      "pesos de rho parcial mediano controlando posicion, 121 videos / 101 canales.",
       "n_videos_ajuste": len(dev), "weights": weights, "calibrator": table}
(ROOT / "models" / FORMULA_FILE).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("guardado", ROOT / "models" / FORMULA_FILE, "| pesos:", len(weights), "| top-10 usados:", len(vals))
