"""Exportación de un clip: corte exacto → reencuadre 9:16 → subtítulos quemados → MP4 H.264/AAC."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.common.logging_utils import get_logger
from src.ingest.media import _require, probe, run
from src.render.captions import build_ass, write_ass
from src.render.reframe import FaceDetector, analyze, render_frames
from src.transcribe.whisper import Word

log = get_logger(__name__)


def export_clip(video: Path, start: float, end: float, words: list[Word], keywords: set[str], lang: str,
                out_path: Path, rcfg: dict, detector: FaceDetector, hook_text: str | None = None) -> dict:
    work = out_path.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    ffmpeg = _require("ffmpeg")
    cut = work / "cut.mp4"
    run([ffmpeg, "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{end - start:.3f}",
         "-vf", f"scale=-2:'min({rcfg.get('max_src_height', 1080)},ih)'", "-r", str(rcfg["fps"]),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k", str(cut)])
    info = probe(cut)
    W, H = rcfg["width"], rcfg["height"]
    if info["has_video"] and info["width"] / info["height"] > W / H + 0.05:
        plan = analyze(cut, detector, rcfg["sample_fps"], rcfg)
        framed = render_frames(cut, work / "framed.mp4", plan, W, H)
        layout = plan.layout
    else:  # ya es vertical (o sin video): sólo escala
        framed, layout = cut, "native"
    ass = write_ass(work / "subs.ass", build_ass(words, start, end, keywords, lang, rcfg["captions"], W, H, hook_text))
    vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,ass=subs.ass"
    # ffmpeg se ejecuta con cwd=work para evitar el escape de rutas de Windows en el filtro ass
    import subprocess

    cmd = [ffmpeg, "-y", "-v", "error", "-i", framed.name if framed.parent == work else str(framed), "-i", "cut.mp4",
           "-map", "0:v:0", "-map", "1:a:0?", "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", "out.mp4"]
    proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg (render final) falló: {proc.stderr[-1500:]}")
    shutil.move(str(work / "out.mp4"), out_path)
    shutil.copy(ass, out_path.with_suffix(".ass"))
    shutil.rmtree(work, ignore_errors=True)
    log.info("clip exportado: %s (layout=%s, %.1fs)", out_path.name, layout, end - start)
    return {"path": str(out_path), "layout": layout}
