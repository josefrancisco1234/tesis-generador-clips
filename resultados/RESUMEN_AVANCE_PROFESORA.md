# Resumen de avance (2026-10-12)

**Sistema:** a partir de un video largo propone clips verticales 9:16 con un score 0-99 de potencial de retencion. **El sistema usa solo el contenido del video** (audio y texto). La curva "Most Replayed" de YouTube se usa unicamente para entrenar y para calificar.

## Datos
- 444 videos de desarrollo con curva de calor (ES y EN, canales distintos; validacion agrupada por canal).
- 93 videos nuevos (cohorte 1) y 63 videos nuevos (cohorte 2) evaluados una sola vez, con hipotesis escritas antes.

## Resultado sin ver la curva (hit@5: algun clip de los 5 toca un pico real)

| | Azar | Formula V0 | V1 (clips <= 45 s) | Aprendizaje R0 | Calor directo (usa curva; techo) |
|---|---|---|---|---|---|
| Cohorte 1 (n = 93) | 27-32 % | 47 % | 55 % | 62 % | 95 % |
| Cohorte 2 (n = 63) | 30 % | -- | -- | 49 % | 98 % |

- La formula (17 pesos, pausas, silencio, variacion de tono) duplica el F1 del azar y es significativa en 352 videos adicionales (p ~ 1e-17).
- Con solo contenido el acierto esta en ~45-60 % segun la cohorte, contra ~30 % del azar.

## Que se intento y no se confirmo
- Variantes de aprendizaje (R4, ensamble E3): mejoran en desarrollo, **no se confirman en datos nuevos** (cohorte 2: E3 +0.0016 de F1, p = 0.58). No se adoptan.
- Tambien negativos: deep learning de secuencia, juez LLM, senal visual gruesa, embeddings de texto/HuBERT, ingenieria de etiqueta, lexico, categoria del video.

## Lectura
- El pico de retencion se parece a una frontera entre segmentos (silencio, pausa, cambio de tono); el modelo capta eso.
- Los picos dependen tambien de factores que no estan en el video (recomendacion, compartidos), lo que explica el techo de contenido (~50-65 %).
- Caso dificil mostrado: video del PPT, pico unico y angosto al final; solo con contenido falla (0/5). Con la curva (no disponible en un video no subido) acierta.

## Proximos pasos
1. Ampliacion de ~200 videos para entrenar (D-071, en cola) y repetir la confirmacion con n mayor.
2. Hipotesis a preregistrar: mezcla de R0 con submodelo de 17 variables (+0.010 F1 en desarrollo, p = 0.015 sin corregir).
3. Interfaz: pegar un link / subir un archivo y obtener los clips (`scripts/interfaz_clips.bat`), modo solo contenido por defecto.

Detalle completo: `docs/DECISIONES.md` (D-001 a D-073) y `EXPLICACION_COMPLETA.md`.
