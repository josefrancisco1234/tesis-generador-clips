"""
PASO 1 — SEGMENTAR: cortar el video en "candidatos" (trocitos de 15 a 90 segundos)
===================================================================================
Codigo real de: src/segment/candidates.py (funcion sentence_candidates)

Este archivo es una COPIA COMENTADA para explicar el flujo paso a paso.
El codigo que realmente corre esta en src/segment/candidates.py — este archivo
NO se ejecuta como parte del pipeline, solo sirve para leer con calma.

Entrada:  una lista de oraciones ya detectadas en la transcripcion (cada una con
          su tiempo de inicio/fin dentro del video).
Salida:   una lista de "Window" (ventanas) = los candidatos a clip.
"""
import numpy as np


def sentence_candidates(sents, min_s, max_s, targets, start_stride=1, max_n=None, seed=0):
    # sents:   lista de oraciones (cada una con .start y .end en segundos)
    # min_s:   duracion minima de un candidato (15 segundos, ver configs/default.yaml)
    # max_s:   duracion maxima de un candidato (90 segundos)
    # targets: duraciones "objetivo" que se intentan alcanzar, p.ej. [20, 40, 60] segundos
    #          (se prueban las 3 por cada punto de inicio, para variar el tamaño de los clips)

    out = set()  # usamos un set() para no repetir el mismo candidato dos veces
    n = len(sents)

    # --- bucle externo: por cada oracion, probamos empezar un candidato ahi ---
    for i in range(0, n, max(1, start_stride)):
        # start_stride=1 significa "probar TODAS las oraciones como posible inicio"
        # (si fuera 2, probaria una oracion si y otra no, para ir mas rapido)

        # --- bucle interno: por cada duracion objetivo (20s, 40s, 60s...) ---
        for target in targets:
            j = i  # j es el indice de la oracion donde probamos TERMINAR el candidato

            # vamos extendiendo el candidato oracion por oracion...
            while j < n and sents[j].end - sents[i].start < target:
                j += 1
                # ...hasta que la duracion acumulada (sents[j].end - sents[i].start)
                # alcance la duracion objetivo que estamos probando (20, 40 o 60s)

            j = min(j, n - 1)  # no pasarnos del final de la transcripcion
            dur = sents[j].end - sents[i].start  # duracion real del candidato armado

            # si nos pasamos del maximo permitido (90s), retrocedemos una oracion
            # a la vez hasta volver a quedar dentro del rango
            while dur > max_s and j > i:
                j -= 1
                dur = sents[j].end - sents[i].start

            # solo lo guardamos si quedo dentro del rango permitido [15s, 90s]
            if min_s <= dur <= max_s:
                # Window guarda: tiempo de inicio, tiempo de fin, indice de oracion
                # inicial (i), indice de oracion final (j+1), y el "tipo" ("sentence"
                # = alineado a oracion, para distinguirlo de las ventanas fijas de 20s)
                out.add(Window(sents[i].start, sents[j].end, i, j + 1, "sentence"))

    # ordenamos los candidatos por tiempo de inicio (y de fin como desempate)
    cands = sorted(out, key=lambda w: (w.start, w.end))

    # si salieron demasiados candidatos (mas de max_n), se hace un muestreo
    # aleatorio PERO con semilla fija (seed=0) para que sea reproducible:
    # correr esto dos veces siempre da los mismos candidatos.
    if max_n and len(cands) > max_n:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(cands), size=max_n, replace=False))
        cands = [cands[k] for k in keep]

    return cands
    # RESULTADO con nuestro video de ejemplo (-Gzz2LZGowU, 58.7 min): 400 candidatos.
    # Cada uno de esos 400 se convierte luego en UNA FILA de la tabla de features
    # (ver 02_featurizar_builder.py).


# ------------------------------------------------------------------------------
# Nota sobre "Window": es una dataclass simple definida en el mismo archivo:
#
#   @dataclass(frozen=True)
#   class Window:
#       start: float   # segundo de inicio en el video
#       end: float     # segundo de fin
#       s0: int        # indice de la primera oracion que cubre
#       s1: int        # indice de la ultima oracion que cubre (exclusivo)
#       kind: str      # "sentence" | "fixed" | "bin"
#
# frozen=True la hace inmutable y "hasheable", por eso puede vivir dentro de un
# set() (para descartar duplicados automaticamente).
# ------------------------------------------------------------------------------
