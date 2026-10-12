"""Sentimiento por oración con un modelo multilingüe (técnica 7, P9)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.common.logging_utils import get_logger

log = get_logger(__name__)
LABELS = ("positive", "neutral", "negative")


@lru_cache(maxsize=1)
def _pipe(model_name: str, device: str = "auto"):
    from transformers import pipeline

    from src.common.devices import resolve_device

    dev = 0 if resolve_device(device) == "cuda" else -1
    log.info("cargando modelo de sentimiento %s (device=%s)", model_name, "cuda" if dev == 0 else "cpu")
    return pipeline("text-classification", model=model_name, top_k=None, device=dev, truncation=True, max_length=128)


def sentence_sentiment(texts: list[str], model_name: str, batch: int = 32, cache: Path | None = None,
                       device: str = "auto") -> np.ndarray:
    """Matriz (n, 3) con probabilidades [pos, neu, neg]."""
    if cache and cache.exists():
        cached = np.load(cache)
        # Mismo chequeo que en keywords/extractor.py::build_index: un cache de
        # otra corrida con distinto numero de oraciones desalinea los indices
        # s0:s1 y produce slices vacios (bug real encontrado 2026-10-01).
        if len(cached) == len(texts):
            return cached
    if not texts:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.zeros((len(texts), 3), dtype=np.float32)
    preds = _pipe(model_name, device)(texts, batch_size=batch)
    for i, p in enumerate(preds):
        d = {x["label"].lower(): x["score"] for x in p}
        out[i] = [d.get(lbl, 0.0) for lbl in LABELS]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out
