"""
PASO 5 — ELEGIR QUE FEATURES USAR: seleccion automatica de grupos (D-027)
===================================================================================
Codigo real de: src/evaluation/experiments.py (dentro de run_fold)

Copia COMENTADA, no se ejecuta. El codigo real vive en src/evaluation/experiments.py.

Problema que resuelve: con ~115 columnas disponibles (K, S, A, H, F, E, L, y
opcionalmente U), meter TODAS siempre resulto PEOR que usar solo un subconjunto
(demasiadas columnas + pocos videos = el modelo memoriza ruido). Pero elegir el
subconjunto "a ojo" mirando que tan bien le fue en el conjunto de prueba seria
TRAMPA (estarias ajustando la respuesta despues de ver el examen).

Solucion: en cada fold de la validacion cruzada, se prueban 7 combinaciones
candidatas y se elige la que mejor NDCG@10 da en un recorte de VALIDACION interno
(nunca en el conjunto de TEST de ese fold).
"""


def run_fold(tables, videos, train_ids, val_ids, test_ids, cfg, seed, ablations=True, force_groups=None):
    # tables:   dict con las tablas "sentence", "bin", "fixed" (los 3 tipos de ventana)
    # train_ids/val_ids/test_ids: listas de video_id para cada parte del fold
    #           (GroupKFold ya se aseguro de que un mismo canal no quede repartido
    #           entre train y test -> ver nota al final de este archivo)

    cand, bins, fixed = tables["sentence"], tables["bin"], tables["fixed"]
    tr, va = sub(cand, train_ids), sub(cand, val_ids)   # sub() = filtrar filas por video_id
    te = {"cand": sub(cand, test_ids), "bins": sub(bins, test_ids), "fixed": sub(fixed, test_ids)}

    groups_all = cfg.features["groups"]          # p.ej. ["K","S","A","H","F","E","L","U"]
    groups = [g for g in groups_all if g != "U"]  # el grupo U (HuBERT) se evalua aparte (D-038)

    params = cfg.ranking["lgbm"]                  # los mismos hiperparametros de siempre

    # --- modelo "de referencia": con TODOS los grupos (sin seleccion) ---
    feats = feature_columns(cand, groups)         # ~90 columnas (todo menos U)
    booster = train_lgbm(tr, va, feats, params, seed)
    # -> esto se guarda como "lgbm_full" para poder comparar

    # --- LA PARTE IMPORTANTE: probar 7 subconjuntos candidatos ---
    subsets = [
        ["A"],                          # solo audio
        ["A", "H"],                     # audio + hook
        ["A", "E"],                     # audio + eventos (risa/aplausos)
        ["A", "L"],                     # audio + linguisticas
        ["K", "A"],                     # keywords + audio
        ["K", "S", "A", "H", "F"],      # casi todo, sin E ni L
        groups,                         # TODOS los grupos (sin U)
    ]
    scored = []  # aqui se van guardando (puntaje, modelo, columnas, nombre) de cada intento

    def try_subset(sset):
        fs = feature_columns(cand, sset)              # columnas de este subconjunto
        b = train_lgbm(tr, va, fs, params, seed)       # se entrena UN modelo nuevo por subconjunto
        # el puntaje que importa es el NDCG@10 calculado sobre "va" (VALIDACION),
        # jamas sobre "te" (TEST) -> asi la eleccion no hace trampa
        score = float(b.best_score.get("val", {}).get(f"ndcg@{params['eval_at'][-1]}", -1.0))
        scored.append((score, b, fs, "+".join(sset)))

    for sset in subsets:
        try_subset(sset)   # se entrenan 7 modelos completos, uno por combinacion

    # se elige el de MEJOR puntaje en validacion (empates se resuelven por orden)
    best, best_feats, best_name = max(scored, key=lambda c: c[0])[1:]

    # ESTE es el modelo que finalmente se evalua contra el TEST del fold, y el
    # que (si gana la mayoria de los 5 folds) termina siendo el modelo desplegado
    # en models/ranker_lgbm.txt
    return best, best_feats, best_name


# ------------------------------------------------------------------------------
# RESULTADO REAL de este proceso (ver docs/DECISIONES.md, D-027 y ranker_meta.json):
#
#   Fold 1: eligio K+S+A+H+F
#   Fold 2: eligio A+E   <- gana
#   Fold 3: eligio A+E   <- gana
#   Fold 4: eligio K+S+A+H+F+E+L
#   Fold 5: eligio A+E   <- gana
#
#   Por MAYORIA (3 de 5 folds), el modelo desplegado usa A+E: 24 columnas
#   (10 de audio puro + 14 de eventos PANNs: risa, aplausos, vitores, musica).
#
#   Esto significa que, con el dataset actual, el texto (keywords, sentimiento)
#   casi no ayuda a predecir que sera "replayed" — lo que manda es COMO suena
#   el audio, no QUE se dice.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# NOTA: por que el split es por CANAL y no por video (regla no negociable,
# ver CLAUDE.md y DECISIONES.md). GroupKFold usa channel_id (columna del CSV
# que vimos al principio) para decidir los grupos:
#
#   from sklearn.model_selection import GroupKFold
#   gkf = GroupKFold(n_splits=5)
#   for train_idx, test_idx in gkf.split(videos, groups=videos["channel_id"]):
#       ...
#
# Si dos videos del mismo canal (mismo presentador, mismo estilo de voz)
# cayeran uno en train y otro en test, el modelo podria "reconocer la voz"
# del presentador en vez de aprender señales genericas de audio/texto que si
# generalicen a presentadores nuevos. Eso seria una fuga de informacion
# silenciosa, parecida a la que ya se detecto una vez en D-018.
# ------------------------------------------------------------------------------
