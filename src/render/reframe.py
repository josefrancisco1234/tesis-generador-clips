"""Reencuadre 16:9 → 9:16 con seguimiento de rostro y hablante activo (técnica 10, P10).

- Detección de rostros: MediaPipe FaceDetector (BlazeFace) con respaldo YuNet (OpenCV).
- Hablante activo: Light-ASD (P10) queda como opción futura por su cadena de preprocesamiento;
  aquí se usa el "plan B" del diseño: actividad de la región de la boca (diferencia entre frames
  muestreados en la mitad inferior del rostro), suavizada en ~1 s. D-012.
- Suavizado: EMA + zona muerta + histéresis mínima para cambiar de hablante.
- Layouts: fill (sigue al rostro), split (2 rostros persistentes), fit (sin rostros: video
  escalado sobre fondo desenfocado), como los layouts públicos de Opus Clip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.common.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Face:
    cx: float
    cy: float
    w: float
    h: float
    score: float


class FaceDetector:
    def __init__(self, kind: str, models_dir: Path):
        self.kind = kind
        self._mp = self._yn = None
        if kind == "mediapipe":
            try:
                from mediapipe.tasks.python import BaseOptions, vision

                opts = vision.FaceDetectorOptions(base_options=BaseOptions(
                    model_asset_path=str(models_dir / "blaze_face_short_range.tflite")), min_detection_confidence=0.5)
                self._mp = vision.FaceDetector.create_from_options(opts)
            except Exception as exc:
                log.warning("MediaPipe no disponible (%s); uso YuNet", exc)
                self.kind = "yunet"
        if self.kind == "yunet":
            self._yn_path = str(models_dir / "face_detection_yunet_2023mar.onnx")

    def detect(self, frame_bgr: np.ndarray) -> list[Face]:
        h, w = frame_bgr.shape[:2]
        if self._mp is not None:
            import mediapipe as mp

            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            res = self._mp.detect(img)
            out = []
            for d in res.detections:
                bb = d.bounding_box
                out.append(Face((bb.origin_x + bb.width / 2) / w, (bb.origin_y + bb.height / 2) / h,
                                bb.width / w, bb.height / h, d.categories[0].score))
            return out
        if self._yn is None or self._yn_size != (w, h):
            self._yn = cv2.FaceDetectorYN.create(self._yn_path, "", (w, h), 0.6)
            self._yn_size = (w, h)
        _, faces = self._yn.detect(frame_bgr)
        if faces is None:
            return []
        return [Face((f[0] + f[2] / 2) / w, (f[1] + f[3] / 2) / h, f[2] / w, f[3] / h, float(f[-1])) for f in faces]


@dataclass
class CropPlan:
    layout: str                 # fill | split | fit
    times: np.ndarray           # segundos relativos al clip
    centers: np.ndarray         # centro x normalizado (fill) por tiempo
    centers_b: np.ndarray | None = None  # segundo rostro (split)


def analyze(video: Path, detector: FaceDetector, sample_fps: float, cfg: dict) -> CropPlan:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / sample_fps)))
    times, faces_t, activity = [], [], []
    prev_gray, idx = None, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = [f for f in detector.detect(small) if f.w > 0.04]
            acts = []
            for f in faces:  # energía de movimiento en la mitad inferior del rostro (boca)
                H, W = gray.shape
                x0, x1 = int((f.cx - f.w / 2) * W), int((f.cx + f.w / 2) * W)
                y0, y1 = int(f.cy * H), int((f.cy + f.h / 2) * H)
                if prev_gray is not None and x1 > x0 and y1 > y0:
                    acts.append(float(np.abs(gray[y0:y1, x0:x1].astype(int) - prev_gray[y0:y1, x0:x1].astype(int)).mean()))
                else:
                    acts.append(0.0)
            times.append(idx / fps)
            faces_t.append(faces)
            activity.append(acts)
            prev_gray = gray
        idx += 1
    cap.release()
    times = np.array(times)
    if len(times) == 0:
        return CropPlan("fit", np.array([0.0]), np.array([0.5]))
    n_faces = np.array([len(f) for f in faces_t])
    if (n_faces > 0).mean() < 0.3:
        return CropPlan("fit", times, np.full(len(times), 0.5))

    # hablante activo: rostro con más actividad (suavizada en ~1 s) ponderada por tamaño
    win = max(1, int(sample_fps))
    centers, current, last_switch = [], None, -1e9
    for i, (t, faces) in enumerate(zip(times, faces_t)):
        if not faces:
            centers.append(centers[-1] if centers else 0.5)
            continue
        scores = []
        for fi, f in enumerate(faces):
            hist = [activity[j][k] for j in range(max(0, i - win), i + 1)
                    for k, g in enumerate(faces_t[j]) if abs(g.cx - f.cx) < 0.08 and k < len(activity[j])]
            scores.append((np.mean(hist) if hist else 0.0) * (0.5 + f.w))
        best = faces[int(np.argmax(scores))].cx
        if current is None or (abs(best - current) > 0.1 and t - last_switch >= cfg["speaker_switch_min_s"]):
            if current is not None and abs(best - current) > 0.1:
                last_switch = t
            current = best
        elif current is not None:
            # sigue al rostro actual si se movió un poco
            near = min(faces, key=lambda f: abs(f.cx - current))
            if abs(near.cx - current) < 0.1:
                current = near.cx
        centers.append(current)
    centers = smooth(np.array(centers), cfg["smoothing_alpha"], cfg["deadzone_frac"])

    two = n_faces == 2
    if cfg.get("layout", "auto") == "split" or (cfg.get("layout", "auto") == "auto" and two.mean() > 0.8 and _switches(centers) > len(times) / (5 * sample_fps)):
        a = np.array([min(f, key=lambda x: x.cx).cx if len(f) == 2 else np.nan for f in faces_t])
        b = np.array([max(f, key=lambda x: x.cx).cx if len(f) == 2 else np.nan for f in faces_t])
        a = pd_ffill(a, 0.3)
        b = pd_ffill(b, 0.7)
        return CropPlan("split", times, smooth(a, 0.1, 0.03), smooth(b, 0.1, 0.03))
    return CropPlan("fill", times, centers)


def _switches(c: np.ndarray) -> int:
    return int((np.abs(np.diff(c)) > 0.08).sum())


def pd_ffill(x: np.ndarray, default: float) -> np.ndarray:
    out, last = x.copy(), default
    for i, v in enumerate(out):
        if np.isnan(v):
            out[i] = last
        else:
            last = v
    return out


def smooth(x: np.ndarray, alpha: float, deadzone: float) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    cur = x[0]
    for i, v in enumerate(x):
        if abs(v - cur) > deadzone:
            cur = cur + alpha * (v - cur)
        out[i] = cur
    return out


def render_frames(src: Path, dst: Path, plan: CropPlan, width: int, height: int) -> Path:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    crop_w = min(W, int(H * width / height))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        if plan.layout == "fit":
            out = _fit(frame, width, height)
        elif plan.layout == "split":
            half_h = height // 2
            ca = float(np.interp(t, plan.times, plan.centers))
            cb = float(np.interp(t, plan.times, plan.centers_b))
            cw = min(W, int(H * width / half_h))
            top = _crop(frame, ca, cw, W)
            bot = _crop(frame, cb, cw, W)
            out = np.vstack([cv2.resize(top, (width, half_h)), cv2.resize(bot, (width, height - half_h))])
        else:
            cx = float(np.interp(t, plan.times, plan.centers))
            out = cv2.resize(_crop(frame, cx, crop_w, W), (width, height), interpolation=cv2.INTER_AREA)
        writer.write(out)
        i += 1
    cap.release()
    writer.release()
    return dst


def _crop(frame: np.ndarray, cx: float, crop_w: int, W: int) -> np.ndarray:
    x0 = int(np.clip(cx * W - crop_w / 2, 0, W - crop_w))
    return frame[:, x0:x0 + crop_w]


def _fit(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    H, W = frame.shape[:2]
    bg = cv2.GaussianBlur(cv2.resize(frame, (width, height)), (0, 0), 25)
    scale = width / W
    fg = cv2.resize(frame, (width, int(H * scale)))
    y0 = (height - fg.shape[0]) // 2
    bg[y0:y0 + fg.shape[0]] = fg
    return bg
