"""
PASO 4 — ENTRENAR: el LightGBM LambdaRank que aprende a ordenar los candidatos
===================================================================================
Codigo real de: src/ranking/model.py (funciones _groups, sort_by_group, train_lgbm)

Copia COMENTADA, no se ejecuta. El codigo real vive en src/ranking/model.py.

Entrada:  la tabla gigante (todos los videos concatenados, ~170,000 filas),
          dividida en "train" (para aprender) y "val" (para saber cuando parar).
Salida:   un "booster" de LightGBM: la formula final (conjunto de arboles) que,
          dado un candidato nuevo, devuelve un puntaje.
"""
import numpy as np
import pandas as pd


def _groups(df: pd.DataFrame) -> np.ndarray:
    """Tamaños de grupo en el orden de aparicion (df debe estar ordenado por video_id)."""
    # Esto es LO MAS IMPORTANTE de todo el entrenamiento y lo mas facil de pasar
    # por alto: LightGBM con objetivo "lambdarank" NO sabe que filas pertenecen
    # a que video a menos que se lo digamos explicitamente con este vector.
    #
    # Ejemplo: si el video A tiene 400 candidatos y el video B tiene 350,
    # _groups(df) devuelve algo como [400, 350, ...]. LightGBM entiende:
    # "las primeras 400 filas son un grupo de ranking (no las compares con
    # nada de fuera), las siguientes 350 son otro grupo", etc.
    return df.groupby("video_id", sort=False).size().to_numpy()


def sort_by_group(df: pd.DataFrame) -> pd.DataFrame:
    # Hay que ordenar la tabla por video_id ANTES de calcular _groups(), porque
    # LightGBM asume que las filas de un mismo grupo estan CONTIGUAS (una
    # detras de otra). Si estuvieran mezcladas, el conteo de _groups() no
    # coincidiria con la realidad y el entrenamiento saldria mal sin avisar.
    return df.sort_values(["video_id", "start"], kind="stable").reset_index(drop=True)


def train_lgbm(train, val, features, params, seed, label="y_grade"):
    import lightgbm as lgb

    # Paso 1: ordenar ambos conjuntos para que _groups() funcione bien
    train, val = sort_by_group(train), sort_by_group(val)

    # Paso 2: separar los parametros "de entrenamiento" (num_boost_round,
    # early_stopping_rounds) de los parametros "del modelo" (objective, etc.)
    p = {k: v for k, v in params.items() if k not in ("num_boost_round", "early_stopping_rounds")}
    p.update(seed=seed, deterministic=True, force_row_wise=True, num_threads=8)
    # seed=42 fijo en todo el proyecto -> correr esto dos veces da EXACTAMENTE
    # el mismo modelo (reproducibilidad, ver docs/DECISIONES.md).

    # Paso 3: armar el "Dataset" de LightGBM
    dtr = lgb.Dataset(
        train[features],          # las ~24 (o las que sean) columnas de features elegidas
        train[label],              # la columna y_grade (0 a 4) -> lo que debe predecir
        group=_groups(train),      # <- el vector magico explicado arriba
        free_raw_data=False,
    )
    dva = lgb.Dataset(val[features], val[label], group=_groups(val), reference=dtr)
    # "reference=dtr" le dice a LightGBM que dva usa el MISMO binning/escalas
    # que dtr, para que la comparacion de validacion sea justa.

    # Paso 4: entrenar de verdad
    booster = lgb.train(
        p,
        dtr,
        num_boost_round=params["num_boost_round"],       # hasta 1000 arboles
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(params["early_stopping_rounds"], verbose=False),
            # se detiene si NDCG en "val" no mejora en 50 rondas seguidas
            # (evita seguir entrenando y memorizar ruido = sobreajuste)
            lgb.log_evaluation(0),  # silencia el log linea por linea
        ],
    )
    # booster.best_iteration = en que arbol #N se quedo (el mejor segun "val")
    return booster


# ------------------------------------------------------------------------------
# ¿QUE HACE "LAMBDARANK" POR DENTRO? (intuicion, sin formulas)
#
# LightGBM arma arboles de decision UNO DETRAS DE OTRO (gradient boosting).
# El primer arbol es simple y comete muchos errores de orden. El segundo arbol
# se entrena mirando ESOS ERRORES especificos (que pares de candidatos quedaron
# mal ordenados dentro de un mismo video) e intenta corregirlos un poco. El
# tercero corrige lo que el segundo no pudo, y asi sucesivamente.
#
# "lambdarank" (a diferencia de una regresion normal) no penaliza por el
# VALOR exacto que predice, sino por el ORDEN: si dentro de un video el
# candidato con y_grade=4 queda, segun el modelo, por debajo del que tiene
# y_grade=1, ESO es lo que se penaliza y se corrige en el siguiente arbol.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# HIPERPARAMETROS REALES usados (configs/default.yaml -> ranking.lgbm):
#
#   objective: lambdarank
#   metric: ndcg
#   eval_at: [5, 10]                 # mide que tan bien ordena el top-5 y el top-10
#   learning_rate: 0.05              # que tan grande es la correccion de cada arbol
#   num_leaves: 31                   # complejidad maxima de cada arbol individual
#   min_data_in_leaf: 20             # minimo de candidatos por hoja (anti-sobreajuste)
#   feature_fraction: 0.8            # cada arbol usa 80% de las columnas, al azar
#   bagging_fraction: 0.8            # cada arbol usa 80% de las FILAS, al azar
#   lambdarank_truncation_level: 20  # solo compara contra los primeros 20 de cada grupo
#   num_boost_round: 1000            # tope de arboles
#   early_stopping_rounds: 50        # se detiene si no mejora en 50 rondas
#
# feature_fraction y bagging_fraction son basicamente "aleatoriedad a proposito":
# si cada arbol viera siempre TODAS las columnas y TODAS las filas, todos los
# arboles saldrian parecidos entre si (poco util). Al variar un poco cada vez,
# el conjunto de arboles se vuelve mas robusto (menos sobreajuste).
# ------------------------------------------------------------------------------
