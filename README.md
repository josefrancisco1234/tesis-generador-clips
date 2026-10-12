# Generador de clips 9:16 con puntaje de retención

Tesis de José Francisco Núñez del Prado Vega · Universidad de Lima.
A partir de un video largo propone clips verticales 9:16, cada uno con un puntaje 0–99. **Elige solo con el contenido**
(audio y texto); la curva "Most Replayed" de YouTube se usa después, solo para calificar.

## Cómo ejecutarlo

```bash
python main.py            # menú
python main.py web        # página web local: http://localhost:8765
python main.py formulas   # puntajes y fórmulas de los 3 videos de ejemplo
python main.py clips https://www.youtube.com/watch?v=l_-igAkbVII
```

Requiere el entorno `C:\tesis_data\.venv` (Whisper, LightGBM, ffmpeg) y los datos en `D:\tesis_data`.

## Los 3 videos de ejemplo

| # | URL | Idioma | Clips sobre un pico real |
|---|---|---|---|
| 1 | https://www.youtube.com/watch?v=l_-igAkbVII | ES | 4/5 |
| 2 | https://www.youtube.com/watch?v=UIC10AA34YU | EN | 3/5 |
| 3 | https://www.youtube.com/watch?v=3KfZ30O2fEQ | EN | 3/5 |

Referencia honesta: en 548 videos nuevos, el método acierta al menos 1 de 5 en ~56 % de los casos (azar ~30 %).
Estos tres son de los mejores casos.

## Estructura

```
main.py                    punto de entrada (menú)
src/                       código del sistema
  pipeline.py              orquesta: descarga -> transcribe -> segmenta -> variables -> puntúa -> recorta
  features/                las 68 variables (builder.py, audio.py, extra.py)
  ranking/retention.py     la fórmula de 17 pesos, el score 0-99 y la explicación
  segment/                 candidatos y NMS (elige 5 sin solapar)
  render/                  recorte 9:16 + subtítulos (ffmpeg)
models/                    formula_calor_D045.json (los 17 pesos) y modelos
scripts/                   web_clips.py (la web), generar_clips.py, interfaz_clips.py, demos
docs/                      MAPA_FORMULAS_EN_CODIGO.md, FORMULAS_Y_VARIABLES.md, EXPLICACION_COMPLETA.md, DECISIONES.md
  codigo_explicado/        05_formulas_como_en_la_web.py: las fórmulas en el orden de la página
resultados/                resumen para la profesora, ejemplos explicados, gráficos
data/manifests/            la lista de videos (youtube_mr.csv) y su estado
```

## Para sustentar el código (5 minutos)

1. `docs/MAPA_FORMULAS_EN_CODIGO.md`: dónde vive cada fórmula (archivo y línea).
2. `docs/codigo_explicado/05_formulas_como_en_la_web.py`: las fórmulas ordenadas como en la web. Ejecutarlo muestra los puntajes.
3. `src/ranking/retention.py`: `formula_raw` (percentil × peso), `calibrate` (0–99), `explain`.
4. `src/features/builder.py`: `window_features`, donde nacen las variables.
