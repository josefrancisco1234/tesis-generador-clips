"""Generación de ventanas candidatas.

- `sentence_candidates`: tramos continuos alineados a oración de 15–90 s (P2 PodReels, P8 SVHighlights).
- `fixed_windows`: ventanas de 20 s (línea base P4 Minimal Clips).
- `bin_windows`: los 100 bins de la curva Most Replayed (protocolo de evaluación de Rhapsody).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.segment.sentences import Sentence


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    s0: int = -1  # primera oración (inclusive), -1 si no aplica
    s1: int = -1  # última oración (exclusive)
    kind: str = "sentence"

    @property
    def duration(self) -> float:
        return self.end - self.start


def sentence_candidates(sents: list[Sentence], min_s: float, max_s: float, targets: list[float],
                        start_stride: int = 1, max_n: int | None = None, seed: int = 0) -> list[Window]:
    out: set[Window] = set()
    n = len(sents)
    for i in range(0, n, max(1, start_stride)):
        for target in targets:
            j = i
            while j < n and sents[j].end - sents[i].start < target:
                j += 1
            j = min(j, n - 1)
            dur = sents[j].end - sents[i].start
            # si se pasó del máximo, retrocede una oración
            while dur > max_s and j > i:
                j -= 1
                dur = sents[j].end - sents[i].start
            if min_s <= dur <= max_s:
                out.add(Window(sents[i].start, sents[j].end, i, j + 1, "sentence"))
    cands = sorted(out, key=lambda w: (w.start, w.end))
    if max_n and len(cands) > max_n:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(cands), size=max_n, replace=False))
        cands = [cands[k] for k in keep]
    return cands


def fixed_windows(duration: float, size: float) -> list[Window]:
    edges = np.arange(0.0, duration, size)
    return [Window(float(a), float(min(a + size, duration)), kind="fixed") for a in edges if min(a + size, duration) - a >= size / 2]


def bin_windows(heatmap: list[dict]) -> list[Window]:
    return [Window(float(b["start_time"]), float(b["end_time"]), kind="bin") for b in heatmap]


def attach_sentences(windows: list[Window], sents: list[Sentence]) -> list[Window]:
    """Asigna s0/s1 a ventanas no alineadas (oraciones cuyo punto medio cae dentro)."""
    mids = np.array([(s.start + s.end) / 2 for s in sents])
    out = []
    for w in windows:
        idx = np.nonzero((mids >= w.start) & (mids < w.end))[0]
        s0, s1 = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (-1, -1)
        out.append(Window(w.start, w.end, s0, s1, w.kind))
    return out


def temporal_iou(a: Window, b: Window) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def overlap_min(a: Window, b: Window) -> float:
    """Intersección sobre la duración del clip más corto (más estricto que IoU)."""
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    m = min(a.duration, b.duration)
    return inter / m if m > 0 else 0.0


def nms(windows: list[Window], scores: np.ndarray, iou: float, top_n: int | None = None, mode: str = "iou") -> list[int]:
    """Supresión de no-máximos temporal: devuelve índices ordenados por score.

    mode="iou" (evaluación) o "iomin" (salida al usuario: evita clips contenidos en otros).
    """
    sim = temporal_iou if mode == "iou" else overlap_min
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        if all(sim(windows[i], windows[k]) <= iou for k in keep):
            keep.append(int(i))
            if top_n and len(keep) >= top_n:
                break
    return keep
