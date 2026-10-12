"""Acceso a Rhapsody (Park et al., COLM 2025) leyendo sólo columnas livianas.

El dataset está 'gated': el autor debe aceptar los términos en
https://huggingface.co/datasets/yhpark/rhapsody y luego ejecutar `hf auth login`.
Se leen únicamente vid, cid, gt, title y transcript (se evitan dva/hubert: ~65 GB).

    python -m src.data.rhapsody --inspect         # muestra el esquema y una fila
    python -m src.data.rhapsody --export          # guarda las columnas livianas en data_root
"""
from __future__ import annotations

import argparse

from src.common.config import load_config
from src.common.logging_utils import get_logger, setup_logging

log = get_logger(__name__)
REPO = "datasets/yhpark/rhapsody"
LIGHT = ["vid", "cid", "plid", "gt", "title", "transcript", "segment_summaries"]


def _files(fs, split: str) -> list[str]:
    return sorted(fs.glob(f"{REPO}/data/{split}-*.parquet"))


def iter_light(split: str):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for f in _files(fs, split):
        with fs.open(f, "rb", block_size=8 * 1024 * 1024) as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                yield pf.read_row_group(rg, columns=LIGHT).to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    setup_logging(cfg.logging["level"], cfg.logging["file"])
    if args.inspect:
        df = next(iter_light("test"))
        row = df.iloc[0]
        print(df.dtypes)
        print("gt:", len(row["gt"]), "transcript (inicio):", str(row["transcript"])[:1500])
    if args.export:
        import pandas as pd

        out = cfg.path("interim") / "rhapsody"
        out.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            parts = list(iter_light(split))
            pd.concat(parts).to_parquet(out / f"{split}.parquet", index=False)
            log.info("rhapsody %s: %d episodios", split, sum(len(p) for p in parts))


if __name__ == "__main__":
    main()
