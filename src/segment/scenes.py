"""Detección de cortes de escena (PySceneDetect) y ajuste de bordes de clips (técnica 9, P8)."""
from __future__ import annotations

from pathlib import Path

from src.common.logging_utils import get_logger

log = get_logger(__name__)


def detect_cuts(video: Path, start: float, end: float, threshold: float = 27.0) -> list[float]:
    """Cortes (en segundos absolutos) dentro de [start, end]. Sólo analiza ese tramo."""
    from scenedetect import ContentDetector, SceneManager, open_video

    try:
        v = open_video(str(video))
        sm = SceneManager()
        sm.auto_downscale = True
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=8))
        v.seek(max(0.0, start))
        sm.detect_scenes(v, end_time=v.base_timecode + end, show_progress=False)
        cuts = [s[0].get_seconds() for s in sm.get_scene_list()[1:]]
    except Exception as exc:  # videos sin pista de video, códecs raros
        log.warning("detección de escenas falló (%s): %s", video.name, exc)
        return []
    return [c for c in cuts if start <= c <= end]


def snap(start: float, end: float, cuts: list[float], tol: float) -> tuple[float, float]:
    """Mueve cada borde al corte más cercano si está a ≤ tol segundos."""
    def nearest(t: float) -> float:
        best = min(cuts, key=lambda c: abs(c - t), default=None)
        return best if best is not None and abs(best - t) <= tol else t

    s, e = nearest(start), nearest(end)
    return (s, e) if e - s > 5 else (start, end)
