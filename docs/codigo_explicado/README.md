# Codigo explicado paso a paso

Estas son **copias comentadas** del codigo real del proyecto, creadas solo para
estudiar el flujo con calma. **No se ejecutan** y **no reemplazan** el codigo
real (que sigue en `src/`). Cada archivo tiene comentarios `#` en cada linea
importante, explicando que pasa y por que.

Leelos en este orden (es el mismo orden del pipeline):

1. [`01_segmentar_candidatos.py`](01_segmentar_candidatos.py) — cortar el video en trocitos de 15-90s
2. [`02_featurizar_builder.py`](02_featurizar_builder.py) — convertir cada trocito en ~115 numeros
3. [`03_etiquetar_labels.py`](03_etiquetar_labels.py) — convertir la curva Most Replayed en la "respuesta correcta"
4. [`04_entrenar_model.py`](04_entrenar_model.py) — el entrenamiento real de LightGBM (`train_lgbm`)
5. [`05_seleccion_grupos_experiments.py`](05_seleccion_grupos_experiments.py) — como se elige que columnas usar (D-027)

El codigo real y sin comentarios didacticos esta en:

- `src/segment/candidates.py`
- `src/features/builder.py`, `src/features/audio.py`, `src/features/extra.py`
- `src/ranking/labels.py`
- `src/ranking/model.py`
- `src/evaluation/experiments.py`

Ver tambien los dos Word generados sobre el mismo tema:
`Del_CSV_al_grafico_Most_Replayed.docx` y `Como_se_alimenta_el_modelo.docx`
(en `Escritorio\activo\` de esta PC).
