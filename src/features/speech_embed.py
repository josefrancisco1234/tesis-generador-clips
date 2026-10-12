"""Grupo U — embeddings de habla autosupervisados (HuBERT), pensados para ejecutarse en GPU.

Rhapsody (Park et al., COLM 2025) reporta que añadir embeddings HuBERT al texto es su mayor mejora
de audio (hit rate 0.452 → 0.490). En CPU el costo medido en el equipo del autor fue de 875 s por
minuto de audio (inviable), por lo que el grupo queda **desactivado por defecto** y se habilita en
`configs/gpu.yaml`.

Procedimiento:
  1. El audio se corta en trozos de `chunk_s` segundos (evita el costo cuadrático de la atención).
  2. Se toma la capa `layer` (10 por defecto, como Rhapsody) y se promedia en el tiempo → 768 dims.
  3. Se ajusta un PCA **sólo con videos de entrenamiento** y se proyecta a `n_components`.
  4. Cada ventana promedia los trozos que la solapan → features `u_00 … u_NN`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger

log = get_logger(__name__)
MODEL = "facebook/hubert-base-ls960"


@lru_cache(maxsize=1)
def _model(name: str, device: str):
    import torch
    from transformers import HubertModel

    from src.common.devices import resolve_device

    dev = resolve_device(device)
    log.info("cargando %s (device=%s)", name, dev)
    # fp32 a propósito: con `.half()` HuBERT devuelve NaN desde la capa 0 (D-035).
    m = HubertModel.from_pretrained(name).eval().to(dev)
    return m, dev, torch


def chunk_embeddings(wav: Path, cache: Path, sr: int = 16000, chunk_s: float = 10.0, layer: int = 10,
                     device: str = "auto", model_name: str = MODEL,
                     batch_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(centros_de_tiempo, embeddings) con un vector por trozo de `chunk_s` segundos.

    Los trozos completos (todos del mismo largo) se procesan en lotes de `batch_size`: no hace falta
    relleno, así que el resultado es el mismo que trozo a trozo. HuBERT base usa normalización de grupo,
    donde el relleno con ceros SÍ altera la salida aunque haya `attention_mask`; por eso el trozo final,
    más corto, se procesa solo. Por defecto 1: en la GTX 1660 SUPER agrupar no acelera y sube la VRAM (D-035).
    """
    if cache.exists():
        d = np.load(cache)
        return d["t"], d["emb"]
    from src.ingest.media import load_wav

    m, dev, torch = _model(model_name, device)
    y = load_wav(wav, sr)
    n = int(chunk_s * sr)
    chunks = [y[i:i + n] for i in range(0, max(len(y) - n // 2, 1), n)]
    valid = [(i, ch) for i, ch in enumerate(chunks) if len(ch) >= sr]  # menos de 1 s: no aporta
    full = [(i, ch) for i, ch in valid if len(ch) == n]
    tail = [(i, ch) for i, ch in valid if len(ch) != n]

    def _run(batch: list[np.ndarray]) -> np.ndarray:
        x = torch.tensor(np.stack(batch), device=dev)
        hs = m(x, output_hidden_states=True).hidden_states[layer]
        return hs.mean(dim=1).float().cpu().numpy()

    got: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for j in range(0, len(full), max(batch_size, 1)):
            part = full[j:j + max(batch_size, 1)]
            for (i, _), v in zip(part, _run([c for _, c in part])):
                got[i] = v
        for i, ch in tail:
            got[i] = _run([ch])[0]
    order = sorted(got)
    emb = np.stack([got[i] for i in order]) if order else np.zeros((0, 768), dtype=np.float32)
    if not np.isfinite(emb).all():  # nunca cachear embeddings corruptos
        raise RuntimeError(f"embeddings no finitos en {wav}: revisar precisión numérica (D-035)")
    t = np.array([i * chunk_s + chunk_s / 2 for i in order], dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=t, emb=emb.astype(np.float32))
    return t, emb


def fit_pca(embeddings: list[np.ndarray], n_components: int, seed: int, out: Path):
    """Ajusta el PCA con embeddings de los videos de ENTRENAMIENTO y lo guarda."""
    from sklearn.decomposition import PCA

    x = np.concatenate([e for e in embeddings if len(e)])
    pca = PCA(n_components=n_components, random_state=seed).fit(x)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, components=pca.components_, mean=pca.mean_,
                        explained=pca.explained_variance_ratio_)
    log.info("PCA ajustado con %d trozos: %d componentes, varianza explicada %.2f",
             len(x), n_components, pca.explained_variance_ratio_.sum())
    return pca


def project(emb: np.ndarray, pca_path: Path) -> np.ndarray:
    d = np.load(pca_path)
    return (emb - d["mean"]) @ d["components"].T


def window_features(t: np.ndarray, proj: np.ndarray, start: float, end: float) -> dict[str, float]:
    m = (t >= start) & (t < end)
    v = proj[m].mean(axis=0) if m.any() else np.full(proj.shape[1], np.nan)
    return {f"u_{i:02d}": float(x) for i, x in enumerate(v)}


def augment_table(df: pd.DataFrame, t: np.ndarray, proj: np.ndarray) -> pd.DataFrame:
    extra = pd.DataFrame([window_features(t, proj, s, e) for s, e in zip(df["start"], df["end"])], index=df.index)
    keep = df.drop(columns=[c for c in df.columns if c.startswith("u_")])
    return pd.concat([keep, extra], axis=1)
