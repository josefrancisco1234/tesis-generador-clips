"""Modelos de ranking de segmentos.

- LightGBM LambdaRank (principal, D-005): grupo = video, relevancia graduada 0..4.
- MLP RankNet (baseline neuronal): pérdida pairwise logística sobre pares del mismo video.
- ScoreCalibrator: score crudo → percentil sobre validación → 0..99 (D-006).
- Desglose Hook/Flow/Value: suma de valores SHAP por grupo de features, también en percentiles.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger
from src.features.builder import DIMENSIONS

log = get_logger(__name__)


def _groups(df: pd.DataFrame) -> np.ndarray:
    """Tamaños de grupo en el orden de aparición (df debe estar ordenado por video_id)."""
    return df.groupby("video_id", sort=False).size().to_numpy()


def sort_by_group(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["video_id", "start"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- LightGBM
def train_lgbm(train: pd.DataFrame, val: pd.DataFrame, features: list[str], params: dict, seed: int,
               label: str = "y_grade"):
    import lightgbm as lgb

    train, val = sort_by_group(train), sort_by_group(val)
    p = {k: v for k, v in params.items() if k not in ("num_boost_round", "early_stopping_rounds")}
    p.update(seed=seed, deterministic=True, force_row_wise=True, num_threads=8)
    dtr = lgb.Dataset(train[features], train[label], group=_groups(train), free_raw_data=False)
    dva = lgb.Dataset(val[features], val[label], group=_groups(val), reference=dtr)
    booster = lgb.train(p, dtr, num_boost_round=params["num_boost_round"], valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False),
                                   lgb.log_evaluation(0)])
    log.info("LightGBM: mejor iteración %d, %s", booster.best_iteration,
             {k: round(v, 4) for k, v in booster.best_score.get("val", {}).items()})
    return booster


# ---------------------------------------------------------------- MLP RankNet
class _Standardizer:
    def __init__(self, x: np.ndarray):
        self.med = np.nanmedian(x, axis=0)
        self.med = np.where(np.isnan(self.med), 0.0, self.med)
        filled = np.where(np.isnan(x), self.med, x)
        self.mu, self.sd = filled.mean(axis=0), filled.std(axis=0) + 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.where(np.isnan(x), self.med, x) - self.mu) / self.sd


@dataclass
class MLPRanker:
    features: list[str]
    hidden: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 40
    pairs_per_group: int = 256
    seed: int = 42

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, label: str = "y_grade") -> "MLPRanker":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        x = train[self.features].to_numpy(dtype=np.float64)
        self.std = _Standardizer(x)
        xt = torch.tensor(self.std(x), dtype=torch.float32)
        y = train[label].to_numpy()
        groups = [np.nonzero(train["video_id"].to_numpy() == v)[0] for v in train["video_id"].unique()]
        layers, d = [], len(self.features)
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        for ep in range(self.epochs):
            self.net.train()
            ii, jj = [], []
            for idx in groups:
                a = rng.choice(idx, self.pairs_per_group)
                b = rng.choice(idx, self.pairs_per_group)
                m = y[a] != y[b]
                ii.append(np.where(y[a] > y[b], a, b)[m])
                jj.append(np.where(y[a] > y[b], b, a)[m])
            ii, jj = np.concatenate(ii), np.concatenate(jj)
            perm = rng.permutation(len(ii))
            total = 0.0
            for s in range(0, len(perm), 1024):
                bi = perm[s:s + 1024]
                diff = self.net(xt[ii[bi]]).squeeze(-1) - self.net(xt[jj[bi]]).squeeze(-1)
                loss = nn.functional.softplus(-diff).mean()  # RankNet: -log σ(s_i - s_j)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(bi)
            if ep % 10 == 0 or ep == self.epochs - 1:
                log.info("MLP RankNet época %d: pérdida %.4f", ep, total / max(len(perm), 1))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.std(df[self.features].to_numpy(dtype=np.float64)), dtype=torch.float32)
            return self.net(x).squeeze(-1).numpy()


# ---------------------------------------------------------------- Calibración y desglose
class ScoreCalibrator:
    def __init__(self, reference: np.ndarray, n_q: int = 101):
        self.q = np.quantile(reference, np.linspace(0, 1, n_q))

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        pct = np.interp(raw, self.q, np.linspace(0, 1, len(self.q)))
        return np.clip(np.round(pct * 99), 0, 99).astype(int)

    def to_list(self) -> list[float]:
        return self.q.tolist()

    @classmethod
    def from_list(cls, q: list[float]) -> "ScoreCalibrator":
        obj = cls.__new__(cls)
        obj.q = np.asarray(q)
        return obj


def dimension_contributions(booster, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Suma de contribuciones SHAP (pred_contrib nativo de LightGBM) por dimensión Hook/Flow/Value."""
    contrib = booster.predict(df[features], pred_contrib=True)[:, :-1]
    out = {}
    for dim, prefixes in DIMENSIONS.items():
        cols = [i for i, f in enumerate(features) if f.startswith(prefixes)]
        out[dim] = contrib[:, cols].sum(axis=1) if cols else np.zeros(len(df))
    return pd.DataFrame(out, index=df.index)


@dataclass
class RankerBundle:
    booster: object
    features: list[str]
    calibrator: ScoreCalibrator
    dim_calibrators: dict[str, ScoreCalibrator]
    meta: dict
    explainer: object | None = None            # modelo con TODOS los grupos, sólo para el desglose
    explainer_features: list[str] | None = None

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.booster.predict(df[self.features], num_iteration=self.booster.best_iteration or None)
        out = pd.DataFrame({"raw": raw, "score": self.calibrator(raw)}, index=df.index)
        exp, feats = (self.explainer, self.explainer_features) if self.explainer is not None else (self.booster, self.features)
        dims = dimension_contributions(exp, df, feats)
        for d in dims:
            out[f"score_{d}"] = self.dim_calibrators[d](dims[d].to_numpy())
        return out

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "ranker_lgbm.txt"), num_iteration=self.booster.best_iteration or None)
        if self.explainer is not None:
            self.explainer.save_model(str(directory / "explainer_lgbm.txt"),
                                      num_iteration=self.explainer.best_iteration or None)
        meta = dict(self.meta, features=self.features, calibrator=self.calibrator.to_list(),
                    explainer_features=self.explainer_features,
                    dim_calibrators={k: v.to_list() for k, v in self.dim_calibrators.items()})
        (directory / "ranker_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "RankerBundle":
        import lightgbm as lgb

        meta = json.loads((directory / "ranker_meta.json").read_text(encoding="utf-8"))
        booster = lgb.Booster(model_file=str(directory / "ranker_lgbm.txt"))
        exp_path = directory / "explainer_lgbm.txt"
        explainer = lgb.Booster(model_file=str(exp_path)) if exp_path.exists() else None
        return cls(booster, meta["features"], ScoreCalibrator.from_list(meta["calibrator"]),
                   {k: ScoreCalibrator.from_list(v) for k, v in meta["dim_calibrators"].items()}, meta,
                   explainer, meta.get("explainer_features"))


def make_bundle(booster, val: pd.DataFrame, features: list[str], meta: dict, top_n: int = 10,
                overlap: float = 0.3, explainer=None, explainer_features: list[str] | None = None) -> RankerBundle:
    """Calibra contra los top-N clips (tras NMS) de cada video de validación (D-019).

    Así el score responde "¿qué tan bueno es este clip comparado con los mejores clips de otros
    videos?" y no satura en 99 como ocurre al comparar contra todos los candidatos.
    """
    from src.segment.candidates import Window, nms

    val = val.reset_index(drop=True)
    raw = booster.predict(val[features], num_iteration=booster.best_iteration or None)
    idx: list[int] = []
    for _, g in val.groupby("video_id"):
        wins = [Window(s, e) for s, e in zip(g["start"], g["end"])]
        keep = nms(wins, raw[g.index.to_numpy()], overlap, top_n, mode="iomin")
        idx += g.index.to_numpy()[keep].tolist()
    ref = val.loc[idx]
    exp, feats = (explainer, explainer_features) if explainer is not None else (booster, features)
    dims = dimension_contributions(exp, ref, feats)
    meta = dict(meta, calibration=f"top-{top_n} por video de validación ({len(idx)} clips)")
    return RankerBundle(booster, features, ScoreCalibrator(raw[idx]),
                        {d: ScoreCalibrator(dims[d].to_numpy()) for d in dims}, meta,
                        explainer, explainer_features)
