"""Featuriza + aumenta en paralelo a la transcripción: procesa cada video apenas existe su transcript.

    python -m src.data.featurize_loop data/manifests/youtube_mr.csv data/manifests/youtube_mr_repurpose.csv
Termina cuando todos los videos se procesaron o tras 90 min sin avances.
Con `--una-vez` hace una sola pasada y se detiene al no haber progreso (para tandas ya descargadas).
"""
import sys
import time

import pandas as pd
import torch

torch.set_num_threads(2)
from src.common.config import load_config  # noqa: E402
from src.common.logging_utils import get_logger, setup_logging  # noqa: E402
from src.data.augment import augment_video  # noqa: E402
from src.data.featurize import featurize_video, features_path  # noqa: E402
from src.data.youtube_dataset import transcript_path  # noqa: E402
from src.transcribe.whisper import Transcript  # noqa: E402

cfg = load_config()
setup_logging("INFO", cfg.logging["file"])
log = get_logger("featurize_loop")
una_vez = "--una-vez" in sys.argv
manifests = [a for a in sys.argv[1:] if not a.startswith("--")] or ["data/manifests/youtube_mr.csv"]
def load_manifests() -> list[dict]:
    from pathlib import Path

    parts = [pd.read_csv(p) for p in manifests if Path(p).exists()]
    return pd.concat(parts).drop_duplicates("video_id").to_dict("records")


m = load_manifests()
done, idle = set(), 0
while len(done) < len(m) and idle < 90:
    m = load_manifests()  # los manifests pueden crecer mientras corre
    progressed = False
    for r in m:
        vid = r["video_id"]
        tp = transcript_path(cfg, vid, "base")
        if vid in done or not tp.exists():
            continue
        tr = Transcript.load(tp)
        if tr.language != r["query_lang"] or len(tr.words) <= 500:
            log.info("descartado %s (idioma %s, %d palabras)", vid, tr.language, len(tr.words))
            done.add(vid)
            continue
        t = time.time()
        if featurize_video(cfg, r, "base"):
            cols = pd.read_parquet(features_path(cfg, "sentence", vid)).columns
            if not any(c.startswith("e_") for c in cols) and not (
                    cfg.path("interim") / "audio_events" / f"{vid}.parquet").exists():
                augment_video(cfg, vid, events=True, stride=3.0)
        done.add(vid)
        progressed = True
        log.info("featurize_loop %s en %.0fs (%d/%d)", vid, time.time() - t, len(done), len(m))
    idle = 0 if progressed else idle + 1
    if not progressed:
        if una_vez:
            break
        time.sleep(60)
log.info("featurize_loop terminado: %d", len(done))
