# Registro de decisiones

Formato: `D-### · fecha · decisión · justificación · alternativas descartadas · estado`

---

### D-001 · 2026-09-21 · Proyecto desde cero (sin legacy)
- **Justificación:** la auditoría (`00_auditoria_codigo_previo.md`) no encontró código previo. No hay nada que mover a `legacy/`.
- **Estado:** vigente.

### D-002 · 2026-09-22 · Entrada: cualquier tipo de video
- **Justificación:** es un requisito del autor ("mándame un video de cualquier cosa"). **Amplía** el dominio de la ficha, que se limitaba a contenido conversacional.
- **Consecuencia:** el modelo se entrena en podcasts (Rhapsody) y la generalización se mide fuera de dominio (Mr. HiSum, SVHighlights).
- **Estado:** vigente; falta alinearlo con la asesora.

### D-003 · 2026-09-22 · Selección de los 10 papers
- **Decisión:** Lotus, PodReels, SPOT, Minimal Clips, Rhapsody, Keyword-DETR, UMT, SVHighlights, Engagement Prediction (ECCV'24) y Light-ASD.
- **Justificación:** cada uno cubre una etapa del pipeline y todos se verificaron con DOI o arXiv. PodReels ocupa la técnica 2, que faltaba.
- **Correcciones a datos previos:**
  - Lotus es de IUI 2025, no de CHI.
  - Mr. HiSum es de NeurIPS 2023, no de 2024.
  - Minimal Clips salió en un workshop (IWSDS 2026).
- **Descartados:**
  - TimeSformer/SPOT completo y TimeChat: requieren GPU.
  - Spotify Podcast Dataset: descontinuado.
  - KVSUM y Summarize-Pro: pasan a referencias de apoyo como fórmulas, no entre los 10 papers.
- **Estado:** aprobado por el autor (2026-09-22).

### D-004 · 2026-09-22 · Dataset principal: Rhapsody (COLM 2025)
- **Justificación:** es el único dataset público con curvas Most Replayed en podcasts largos, con IDs y transcripciones WhisperX. Evita transcribir unas 6 000 h en CPU.
- **Alternativas:**
  - Mr. HiSum: videos de 2–5 min con sesgo visual. Queda para evaluación.
  - QVHighlights: saliencia condicionada a una consulta. Queda para evaluación.
- **Estado:** aprobado por el autor (2026-09-22); **en espera** por acceso restringido → ver D-009.

### D-005 · 2026-09-22 · Modelo: LightGBM LambdaRank + baseline MLP RankNet
- **Justificación:** entrena en CPU, es interpretable con SHAP y optimiza el ranking. El baseline neuronal cubre la línea "deep learning / pairwise ranking" de la ficha.
- **Alternativa descartada:** transformer de video (inviable sin CUDA).
- **Estado:** aprobado por el autor (2026-09-22).

### D-006 · 2026-09-22 · Score 0–99 con desglose Hook/Flow/Value, sin Trend
- **Justificación:** replica la interfaz pública de Opus Clip, pero auditable. El score es un percentil calibrado y el desglose sale de SHAP agrupado. Trend no tiene etiqueta pública ni es reproducible.
- **Estado:** aprobado por el autor (2026-09-22).

### D-007 · 2026-09-22 · El LLM solo trabaja sobre el top-N; proveedor configurable con modo `none`
- **Justificación:** controla el costo y la reproducibilidad. PodReels y Lotus mandan todo el contenido al LLM; aquí no.
- **Detalles:** caché en disco y temperatura 0.
- **Estado:** aprobado por el autor (2026-09-22).

### D-008 · 2026-09-22 · El score se define como "potencial de retención", no como "viralidad"
- **Justificación:** Most Replayed mide re-reproducción (P9 define el engagement como retención). Llamarlo "viralidad" sin datos de TikTok o Reels sería una afirmación no validada.
- **Estado:** aprobado; afecta el título de la tesis.

### D-009 · 2026-09-22 · Dataset propio YT-MR (yt-dlp) como fuente principal mientras Rhapsody esté bloqueado
- **Contexto:** Rhapsody es *gated* en Hugging Face; requiere que el autor acepte los términos y haga `hf auth login` (no se pueden ingresar credenciales por él).
- **Decisión:** construir un dataset propio con la curva Most Replayed obtenida vía `yt-dlp` (campo `heatmap`, 100 bins). 71 videos hablados (40 ES, 31 EN), 57 h, 58 canales, máx. 2 videos por canal; **válidos tras la adquisición: 62 videos, 50.6 h, 51 canales (37 ES, 25 EN)**. Manifest versionado en `data/manifests/youtube_mr.csv`.
- **Legal/ética:** sólo IDs y datos derivados se versionan; el audio original se borra tras convertirlo a wav y puede borrarse tras extraer features; uso académico no comercial.
- **Rhapsody:** el lector (`src/data/rhapsody.py`) queda listo para experimentos complementarios cuando haya acceso.
- **Estado:** vigente.

### D-010 · 2026-09-22 · Normalización de la fórmula KVSUM sobre los términos del pasaje
- **Justificación:** si la raíz de la Ec. 1 sumara sobre los pasajes j para un término fijo i, el factor log(N/PF_i) aparecería en numerador y denominador y se cancelaría (el IDF no tendría efecto). Se interpreta como normalización coseno de Salton sobre los términos del pasaje j. Cubierto por `tests/test_keywords.py`.
- **Estado:** vigente.

### D-011 · 2026-09-22 · Validación cruzada agrupada por canal (k=5) además del split fijo 70/15/15
- **Justificación:** con 71 videos, un único test de ~11 videos da métricas muy ruidosas. GroupKFold por canal evita fuga (mismo presentador en train y test) y permite reportar media ± desviación. El modelo final se entrena con el split fijo (semilla 42) y se calibra en validación.
- **Estado:** vigente.

### D-012 · 2026-09-22 · Hablante activo por actividad de boca (plan B) en lugar de Light-ASD
- **Justificación:** Light-ASD requiere su propia cadena de detección y tracking (S3FD + recortes de 112 px + MFCC) y no tiene paquete instalable; integrarlo en Windows/CPU no es trivial. Se implementó el plan B previsto en el diseño: diferencia entre frames en la mitad inferior de cada rostro, suavizada ~1 s, con histéresis de 1.2 s. Light-ASD queda como mejora futura; P10 sigue siendo la referencia de la técnica.
- **Estado:** vigente.

### D-013 · 2026-09-22 · Whisper `base` para el dataset y `small` para inferencia
- **Medición en el equipo del autor (Ryzen 7 7735HS, int8, 8 hilos):** base RTF 0.076 (1 h en ~4.6 min), small RTF 0.215 (1 h en ~13 min), sobre un audio de 15 min.
- **Justificación:** 57 h de dataset con small tomarían ~12 h; con base ~4.6 h (dos workers en paralelo). En inferencia la calidad de subtítulos importa más, por eso small.
- **Estado:** vigente.

### D-014 · 2026-09-22 · Entorno Python 3.11.9 en `C:\tesis_data\.venv` (fuera de OneDrive)
- **Justificación:** pedido del autor (3.11); los datos (audio, transcripciones, features) y el entorno se ubican fuera de OneDrive para no saturar la sincronización. En el repo sólo quedan código, configs, manifests, modelos livianos y reportes.
- **Estado:** vigente.

### D-015 · 2026-09-22 · LLM por defecto Claude Haiku 4.5, con caída a plantillas si no hay API key
- **Contexto:** no hay `ANTHROPIC_API_KEY` ni `OPENAI_API_KEY` en el equipo.
- **Consecuencia:** el pipeline corre con plantillas deterministas (título, hook, estilo, autocontenido por reglas). La ablación "etiqueta LLM top-K (P4)" queda pendiente hasta tener una key; el código (`llm_select_topk`) está listo.
- **Estado:** vigente.

### D-016 · 2026-09-22 · El grupo visual (V) no entra al modelo entrenado
- **Justificación:** para el dataset sólo se descarga audio (≈20 MB/h vs ≈300 MB/h de video). Lo visual se usa en generación (cortes de escena, reencuadre, layouts), no en el ranking. Limitación declarada en la auditoría.
- **Estado:** vigente.

### D-017 · 2026-09-22 · Transcripción propia en vez de subtítulos automáticos de YouTube
- **Contexto:** la descarga de subtítulos automáticos devolvió HTTP 429 (rate limit).
- **Decisión:** transcribir todo con faster-whisper, lo que además da timestamps por palabra consistentes entre entrenamiento e inferencia.
- **Estado:** vigente.

### D-018 · 2026-09-22 · Novedad léxica local (300 s) en lugar de global — corrección de una fuga
- **Hallazgo:** `k_novelty` (fracción de términos nunca vistos antes en el video) correlacionaba −0.60 con la posición del clip: al inicio del video todo es "nuevo". Era la feature más importante del modelo y le filtraba la posición, que ya se había decidido excluir.
- **Corrección:** un término es "visto" si apareció en los 300 s previos (`local_seen`, cubierto por tests). Se re-featurizaron los 62 videos y se repitieron todos los experimentos.
- **Impacto:** P@1 en CV bajó de 0.173 a 0.049 y MRR de 0.268 a 0.170. Los números previos no deben citarse.
- **Estado:** vigente.

### D-019 · 2026-09-22 · Calibración del score contra los top-10 clips por video de validación + NMS de salida por solapamiento sobre el clip más corto
- **Problema:** calibrar contra todos los candidatos hacía que los mejores clips de cualquier video recibieran 99. Además, NMS por IoU ≤ 0.5 permitía entregar un clip contenido dentro de otro.
- **Decisión:** el score 0–99 es el percentil del clip frente a los top-10 clips (tras NMS) de cada video de validación ("¿qué tan bueno es comparado con los mejores clips de otros videos?"). En la salida, se descartan clips que compartan > 30 % del más corto. La evaluación sigue usando IoU 0.5.
- **Estado:** vigente.

### D-020 · 2026-09-22 · Se mantienen los hiperparámetros definidos a priori
- **Justificación:** una configuración más regularizada (`reports/regularized/`) no mejoró de forma consistente. Elegir la configuración después de ver la CV inflaría los resultados sin un test independiente. Queda documentada como análisis de sensibilidad.
- **Estado:** vigente.

---
## Iteración 2 (2026-09-22)

### D-021 · 2026-09-22 · Protocolo desarrollo/confirmación y ampliación del dataset (v2)
- **Decisión:** el dataset v1 (62 videos, `data/manifests/youtube_mr_status_v1.csv`) pasa a ser el **conjunto de desarrollo**: todas las variantes de la iteración 2 se comparan sólo allí (CV por canal + Wilcoxon, `src/evaluation/iterate.py`). Los videos **nuevos** de v2 (consultas ampliadas a 26 ES / 24 EN por género y país, meta 120/120) forman un **conjunto de confirmación** que no participa en ninguna decisión.
- **Justificación:** con 62 videos, elegir entre variantes por CV sobreajusta la selección (garden of forking paths). Un conjunto nuevo e independiente permite confirmar o refutar la mejora.
- **Estado:** vigente.

### D-022 · 2026-09-22 · Hipótesis preregistrada: features relativas al video (`__rk`)
- **Resultado exploratorio en v1 (CV, n=60–61 videos):** `rel_rk` (percentil de cada feature dentro del video) NDCG@10 0.566→0.632 (p=0.071), MRR 0.172→0.220 (p=0.76), F1 temporal 0.086→0.096 (p=0.31). `rel_cx` (contraste local ±90 s) y la combinación no mejoran de forma consistente; las etiquetas alternativas (máximo de la curva, 10 grados) no mejoran.
- **Hipótesis a confirmar en los videos nuevos de v2, fijada antes de verlos:** entrenado en v1 y evaluado en v2-nuevos, `rel_rk` supera a `base` en NDCG@10 (Wilcoxon pareado por video, unilateral, α=0.05). Métricas secundarias: tF1@5, P@5, MRR.
- **Estado:** preregistrada (pendiente de confirmación).

### D-023 · 2026-09-22 · Repurpose-10K como segunda etiqueta (clips humanos)
- **Decisión:** ~80 videos en inglés de Repurpose-10K (Wu et al., AAAI 2025; MIT) que además tienen curva Most Replayed, elegidos al azar (semilla 42) entre los de 20–75 min, máx. 2 por canal. Para ellos se guarda `y_human_overlap` (fracción del candidato cubierta por clips humanos) y `y_human` (≥ 0.5).
- **Uso:** (a) acuerdo entre la re-reproducción del público y la curación humana; (b) evaluar si el ranker recupera los clips humanos sin haber sido entrenado con ellos.
- **Estado:** vigente.

### D-024 · 2026-09-22 · Nuevos grupos de features E (eventos de audio) y L (lingüísticos de engagement)
- **E:** PANNs CNN14, 1 s cada 3 s (RTF medido ≈0.12 por segundo analizado); risa, aplausos, vítores, música y habla por candidato, más risa al final del clip y justo después.
- **L:** Reddy et al. (ACL 2021).
- **Descartados por costo:** HuBERT (875 s por 1 min de audio en CPU compartida) y eGeMAPS.
- **Evaluación:** entran como variantes (+E, +L) en el protocolo desarrollo/confirmación de D-021; no se adoptan sin evidencia.
- **Estado:** vigente.

### D-025 · 2026-09-22 · Dispositivo configurable (CPU/GPU) y guía de migración
- **Decisión:** `transcribe.device` y `features.device` aceptan `auto | cpu | cuda`; `compute_type: auto` elige int8 en CPU, `int8_float32` en GPU Pascal (GTX 10xx, sin fp16 acelerado) y `float16` en RTX 20xx o posterior. `scripts/check_env.py` verifica el entorno y `docs/SETUP_GPU.md` documenta la migración.
- **Justificación:** el autor dispone de otra PC con GPU NVIDIA. La transcripción y las features de audio son el cuello de botella; con GPU se puede subir Whisper de `base` a `small`/`medium` y volver viable HuBERT (descartado en D-024 por costo en CPU).
- **Aviso registrado:** la descarga desde YouTube debe hacerse desde una IP doméstica; las IP de datacenter reciben el bloqueo anti-bot.
- **Estado:** vigente.

### D-026 · 2026-09-22 · Diagramas del sistema versionados como fuente Mermaid
- **Decisión:** `docs/diagramas/*.mmd` (pipeline de inferencia, datos y etiquetas, protocolo experimental) con un visor generado por `scripts/render_diagramas.py` que permite exportar cada diagrama en SVG.
- **Justificación:** las figuras de la tesis deben poder regenerarse y editarse; Mermaid es texto, así que versiona bien y se actualiza junto con el código.
- **Estado:** vigente.

### D-027 · 2026-09-22 · Selección de grupos de features DENTRO de cada fold; el modelo desplegado usa la elección mayoritaria
- **Problema:** con los 7 grupos activos, el modelo completo quedó por debajo de variantes más simples (quitar keywords mejoraba NDCG con p=0.02; "sólo audio" era el mejor en todas las métricas). Elegir "sólo audio" tras ver la CV sería seleccionar sobre el test.
- **Decisión:** en cada fold se entrenan 7 subconjuntos candidatos ([A], [A,H], [A,E], [A,L], [K,A], [K,S,A,H,F], todos) y se elige por NDCG@10 **en la validación interna**. Lo que se reporta es el desempeño de ese *procedimiento* (`lgbm_autogroups`), no de un subconjunto elegido a mano.
- **Modelo desplegado:** el subconjunto elegido por mayoría entre los 5 folds (resultado: **A**, sólo audio, 10 features, elegido en 3/5 folds; los otros fueron A+L y K+A).
- **Desglose Hook/Flow/Value:** se calcula con un modelo "explicador" entrenado con todos los grupos (`models/explainer_lgbm.txt`). Es descriptivo y no decide el ranking; así el producto conserva la explicación aunque el ranker use sólo audio.
- **Estado:** vigente.

### D-028 · 2026-09-22 · Hallazgo: Most Replayed y curación humana miden cosas distintas
- **Evidencia (11 videos de Repurpose-10K con ambas etiquetas):** correlación de Spearman media entre la curva y el solapamiento con clips humanos de **−0.04** (mediana 0.01), y F1 temporal entre los bins highlight y los clips humanos de **0.067**.
- **Consecuencia:** ningún modelo entrenado con Most Replayed recupera los clips humanos mejor que el azar (tF1 0.107 del modelo vs 0.152 del azar; p=0.95). La etiqueta que optimizamos **no** es la que usa un editor.
- **Implicancia para la tesis:** debe reportarse como limitación central y como justificación de la línea futura: entrenar directamente con los ~8.4k videos de Repurpose-10K (clips humanos) y comparar ambas etiquetas.
- **Cautela:** sólo 11 videos; el resto de Repurpose no se pudo descargar hoy por el bloqueo anti-bot de YouTube.
- **Estado:** vigente.

---
## Preparación para GPU (2026-09-24)

### D-029 · 2026-09-24 · Grupo U (embeddings de habla HuBERT) implementado pero desactivado en CPU
- **Decisión:** `src/features/speech_embed.py` calcula embeddings HuBERT por trozos de 10 s (capa 10, como Rhapsody), ajusta un PCA **sólo con videos de entrenamiento** y proyecta a 32 componentes (`u_00…u_31`). Se construye con `scripts/build_speech_embeddings.py`.
- **Estado del grupo:** desactivado por defecto; se habilita con `configs/gpu.yaml`. En CPU el costo medido fue de 875 s por minuto de audio.
- **Justificación:** Rhapsody reporta que es su mayor mejora de audio (hit rate 0.452 → 0.490). Dejarlo listo evita rehacer trabajo al migrar.
- **Nota metodológica:** el PCA es no supervisado, pero igual se ajusta sólo con train para no usar información del test.
- **Estado:** vigente.

### D-030 · 2026-09-24 · Perfiles de configuración con herencia (`extends`)
- **Decisión:** `load_config` admite la clave `extends`, de modo que `configs/gpu.yaml` sólo redefine lo que cambia (dispositivo, modelos de Whisper, grupo U, metas del dataset) y hereda el resto de `configs/default.yaml`. Cubierto por una prueba automatizada.
- **Justificación:** duplicar la configuración completa por máquina garantiza que ambas se desincronicen.
- **Estado:** vigente.

### D-031 · 2026-09-24 · Entrenamiento con etiqueta humana listo, a la espera de datos
- **Decisión:** `src/evaluation/human_train.py` entrena y compara tres variantes sobre los videos con curación humana: modelo entrenado con clips humanos, modelo entrenado con Most Replayed y azar; mide recuperación de clips humanos con validación cruzada por canal y Wilcoxon.
- **Requisito:** al menos 25 videos con `y_human`; hoy hay 12 y la descarga continúa. El script se niega a ejecutarse por debajo de ese umbral en lugar de producir un resultado sin poder estadístico.
- **Estado:** pendiente de datos.

### D-032 · 2026-09-24 · Traslado entre máquinas con herramientas libres, y panel de estado
- **Traslado:** **Syncthing** (MPL-2.0) sincroniza el repo y `C:/tesis_data` de forma continua entre las dos PC, sin nube ni cuenta, con `.stignore` que excluye entorno, cachés y audio; **croc** (MIT) cubre envíos puntuales. `scripts/pack_for_gpu.py` empaqueta lo que vive fuera del repo en `C:/tesis_data/transfer` y `scripts/unpack_transfer.py` lo restaura. El paquete básico pesa 121 MB comprimido. El audio (12 GB en WAV) es opcional: `--audio opus` lo reduce a ~1 GB y se reconvierte a WAV 16 kHz al restaurar. Sin audio, la otra máquina puede reentrenar y evaluar, pero no recalcular features acústicas ni construir el grupo U.
- **Justificación:** el autor pidió explícitamente no depender de OneDrive y usar software libre. Syncthing y croc no requieren cuenta, servidor ni servicio de terceros; el código puede además versionarse con git contra un remoto propio (Forgejo, Gitea o un repositorio bare en disco). Documentado en `docs/TRANSFERENCIA.md`.
- **Panel:** `scripts/build_dashboard.py` genera `docs/panel.html` a partir de los archivos reales (manifests, reports/, outputs/, logs). Si un reporte no existe, la tarjeta lo declara pendiente en lugar de mostrar un número inventado.
- **Estado:** vigente.

---
## Migración a GPU (2026-09-24)

### D-033 · 2026-09-24 · Cambio de ASR (Whisper base → small): todo o nada, una sola vez, al cerrar la adquisición
- **Contexto:** `configs/gpu.yaml` subía `model_dataset` a `small`, pero la laptop sigue transcribiendo con `base` (149 transcripciones al momento de esta decisión, contra 124 en el paquete recibido, que es una foto de las 18:52). Retranscribir ahora con `small` dejaría el dataset mezclado igual.
- **Decisión:** `model_dataset` se mantiene en **`base` en ambas máquinas** (`configs/gpu.yaml` fijado a `base`). No se borra `data/processed/features/`. Secuencia:
  1. Ahora: adquisición y featurización continúan con `base`; aquí se avanza con lo que no depende de la transcripción (grupo U, HuBERT por trozo).
  2. Cuando la laptop termine de descargar, envía un paquete incremental y se consolida todo en la PC con GPU.
  3. Sólo entonces: barrido único con `small` sobre **todos** los videos, borrar `features/`, re-featurizar, reentrenar y anotar en `02_auditoria_modelo.md` el paso de `base` a `small` con su impacto en las métricas, **también si es negativo**.
- **Justificación:** el cambio de ASR altera las transcripciones y con ellas todas las features de texto (K, S, H, F, L); mezclar dos ASR introduciría un confundidor entre videos. Hacerlo una vez, con los datos en una sola máquina, lo aísla.
- **Audio para el barrido:** el audio desde la laptop se envía en **FLAC sin pérdida**; un primer envío en opus a 32 kbps se descartó porque la compresión con pérdida sería un segundo confundidor sobre el cambio de ASR.
- **Alternativas descartadas:** (A) retranscribir ya con `small` (dataset mezclado mientras la laptop siga con `base`); (B) quedarse en `base` definitivamente (desaprovecha la GPU y deja el ASR más débil para la tesis).
- **Estado:** vigente; el paso 3 queda pendiente del cierre de la adquisición.

### D-034 · 2026-09-24 · Grupo U: los embeddings HuBERT por trozo son independientes del ASR
- **Decisión:** el grupo U se construye ahora. Los embeddings por trozo (`interim/speech_emb/*.npz`) dependen sólo del audio, así que se conservan al retranscribir en D-033; sólo hay que rehacer la proyección PCA a columnas `u_*`, que es barata.
- **~~Medición inicial (5,6 s/min, 380 MB)~~ INVÁLIDA:** se hizo con el código en fp16, que devolvía NaN (ver D-035). Se conserva tachada como registro; la medición válida está en D-035.
- **Datos:** `C:\tesis_data` vive físicamente en `D:\tesis_data` (junction desde C:, por espacio en disco); ninguna ruta del código cambia.
- **Estado:** vigente; ejecución completa pendiente de los WAV.

### D-035 · 2026-09-24 · HuBERT en fp32 (el fp16 devolvía NaN) y el agrupado por lotes no acelera
- **Hallazgo (negativo, fuga de calidad silenciosa):** `speech_embed.py` cargaba HuBERT con `.half()` en CUDA. En la GTX 1660 SUPER todas las capas (desde la 0) salen **NaN con cualquier entrada**, incluso silencio; en fp32 son finitas. Sin la prueba, el grupo U se habría construido con embeddings todos NaN, y la primera medición de velocidad (D-034) medía un cálculo vacío. La descubrió una comparación de embeddings entre modos, no las pruebas previas.
- **Corrección:** HuBERT corre en fp32; `chunk_embeddings` rechaza (y no cachea) embeddings no finitos. Cubierto por `tests/test_speech_embed.py`.
- **Hipótesis del autor probada:** agrupar trozos por pasada aceleraría mucho. **Refutada** en esta tarjeta (HuBERT base fp32, 5 min de audio sintético, modelo cargado, mediana de una corrida por configuración):

  | batch_size | s por min de audio | VRAM pico |
  |---|---|---|
  | 1 (original) | 0,34 | 519 MB |
  | 4 | 0,42 | 1 180 MB |
  | 16 | 0,34 | 2 495 MB |
  | 32 | 0,52 | 4 339 MB |

  Un trozo de 10 s ya satura la GPU. Los embeddings son idénticos entre modos (diferencia máxima 0; se agrupan sólo trozos de largo completo, sin relleno, porque HuBERT base usa normalización de grupo y el relleno con ceros altera la salida aunque haya `attention_mask`). Por defecto `batch_size: 1`; el código por lotes queda disponible y probado.
- **Costo real:** 0,34 s por minuto de audio frente a 875 s en CPU (~2 500 veces menos). Para ~100 h de audio: ~35 min. Medido con audio sintético: velocidad y memoria, no calidad.
- **Infraestructura:** caché de Hugging Face movida a `D:	esis_data\hf_cache` (variable de usuario `HF_HOME`) para no llenar C:.
- **Estado:** vigente.
- **Repo canónico (2026-09-24):** desde D-035 el repositorio de la PC con GPU es el **canónico**; la laptop envía sólo datos, nunca código. La laptop conserva la versión con `.half()`: no debe usarse para el grupo U.

### D-036 · 2026-09-24 · Whisper en la GTX 1660 SUPER: fp16 no falla pero es lento; faltaba cargar las DLL CUDA
- **Motivo:** tras D-035 se sospechó que faster-whisper sufriera el mismo problema de fp16. Se validó antes del barrido de D-033.
- **Método:** 2 fragmentos reales de 3 min (ES y EN, minuto 10 de dos videos), mismos parámetros que producción (beam 1, VAD, timestamps por palabra), una corrida por configuración, comparando cada tipo de cómputo contra `float32`. Muestra pequeña: valida ausencia de corrupción, no ranking fino de calidad.
- **Resultado (modelo `small`, texto distinto vs fp32 / tiempo para 3 min de audio):**

  | compute_type | ES | EN | tiempo |
  |---|---|---|---|
  | float32 (referencia) | – | – | 9,6 s / 8,7 s |
  | int8_float32 | 1,5 % | 3,0 % | 9,7 s / 9,0 s |
  | float16 | 1,7 % | 0,0 % | 17,9 s / 17,7 s |

  Con `base`: fp16 0,0 % y 0,4 %, int8_float32 1,1 % y 1,7 %; timestamps medios de int8_float32 hasta 0,06 s de diferencia (máximo 1,5 s en una palabra). **Ninguna variante produjo NaN, texto vacío ni repeticiones anómalas.** La hipótesis "fp16 corrompe Whisper" queda **refutada**; fp16 es correcto pero ~2× más lento que fp32 porque la GTX 16xx no tiene tensor cores.
- **`medium` (inferencia), 1 fragmento de 3 min, incl. carga:** float32 147 s y 5,3 GB de VRAM (con `small` aún en caché), float16 56 s / 2,5 GB, int8_float32 **26 s / 1,6 GB**. fp32 es inviable para `medium`.
- **Decisión:**
  1. `compute_type: auto` devuelve **float32** en las GTX 16xx (regla en `devices.py`, con prueba). Para el barrido de D-033 con `small` es la mejor opción: mismo tiempo que int8 y sin cuantización.
  2. Para `medium` en inferencia conviene `int8_float32` explícito; queda como decisión abierta del autor (no afecta a los experimentos).
  3. El ASR de cada etapa queda definido por modelo **y** tipo de cómputo: el dataset actual se transcribió en la laptop con `base` en int8 (CPU); el barrido de D-033 usará `small` en float32.
- **Fallo encontrado de paso (bloqueante):** faster-whisper en GPU fallaba con `Library cublas64_12.dll is not found`: con `device` y `compute_type` explícitos nunca se importaba `torch`, que es quien registra sus DLL CUDA. `check_env.py` decía "Listo" sin detectarlo. Corregido importando `torch` en `_model` cuando el dispositivo es CUDA. `check_env.py` **no** prueba una inferencia real de CTranslate2; queda pendiente.
- **PANNs y modelo de sentimiento:** no hay `.half()`, `float16` ni `autocast` en `src/` (revisión de código, no ejecución).
- **Estado:** vigente.


### D-037 · 2026-09-24 · Dos rutas de Whisper a propósito (dataset en float32, inferencia en int8_float32) y prueba real de inferencia en `check_env`
- **Decisión:**
  1. **Dataset/experimentos** (`transcribe.compute_type`, `auto` → float32 en la 1660 SUPER): consistencia experimental; es la referencia sin cuantización y el barrido de D-033 con `small` corre así.
  2. **Inferencia de producto** (`transcribe.compute_type_inference: int8_float32` en `configs/gpu.yaml`, usado por `src/pipeline.py` con `model_inference: medium`): 26 s contra 147 s y 1,6 GB contra 5,3 GB de VRAM en el fragmento de 3 min de D-036. Para subtítulos la pérdida por cuantización (0–3 % de palabras distintas en la muestra de D-036, sin medir su efecto en subtítulos) no compensa ese costo.
  Son dos rutas distintas a propósito. Sin la clave `compute_type_inference`, la inferencia usa `compute_type` como antes.
- **`scripts/check_env.py`:** ahora transcribe ~3 s (tono con ruido, modelo `tiny`) por el mismo camino del pipeline, en el dispositivo y con los tipos de cómputo del perfil (`configs/gpu.yaml` si hay GPU; `--config` para otro). Si falla, es **crítico**. Motivo: el verificador decía «Listo» con `cublas64_12.dll` ausente (D-036). Comprobado en positivo (GPU) y en negativo con un `compute_type` inválido, que se reporta como crítico; no se simuló específicamente la ausencia de la DLL, aunque falla en el mismo punto (la codificación, cuyos segmentos la prueba consume).
- **Correcciones menores:** la fila «GPU NVIDIA» usa `resolve_compute_type` en vez de una regla codificada (mostraba float16 para la 1660 SUPER); el espacio libre se mide en la unidad real de los datos (sigue la junction a D:) y ya no da un aviso engañoso.
- **Estado:** vigente. Ojo al cambiar el ASR: la transcripción de prueba usa `tiny` y no afecta a los datos.

### D-038 · 2026-09-24 · Preregistro del grupo U (HuBERT): qué se prueba, con qué datos y qué cuenta como confirmación
**Escrito ANTES de correr `run_all.py` con U y sin haber visto ninguna métrica que incluya U.**
- **Datos:** *desarrollo* = los 121 videos con features actuales (validación cruzada por canal, k=5, semilla 42, hiperparámetros sin tocar, D-020). *Confirmación* = los videos que lleguen en el paquete incremental de la laptop y **no** estén en esos 121; el modelo se entrena con los 121 y se evalúa en ellos, sin mirarlos antes. Salvedad: los 39 videos que confirmaron D-022 ya forman parte del desarrollo, así que no sirven como confirmación de U.
- **Procedimientos fijados a priori** (`src/evaluation/experiments.py`):
  - **Base:** `lgbm_autogroups` = D-027 sin cambios (7 subconjuntos sobre los grupos K, S, A, H, F, E, L). `lgbm_full` sigue siendo los 7 grupos sin U.
  - **Con U:** `lgbm_autogroups_U` = los mismos 7 subconjuntos **más 4**: `[U]`, `[A,U]`, `[A,H,U]` y todos los grupos con U. Ambos eligen por NDCG@10 en la validación interna de cada fold. Complementarias: `lgbm_full_U` y `only_U`.
  - Con el perfil GPU, `lgbm_full` no debe cambiar de significado: por eso el código separa `groups` (sin U) de `groups_all`.
- **Hipótesis primaria (H-U1):** en la confirmación, `lgbm_autogroups_U` supera a `lgbm_autogroups` en **NDCG@10** (Wilcoxon pareado por video, unilateral, α = 0,05). Es la única con decisión asociada. Secundarias, sin corrección por comparaciones múltiples y marcadas como tales: P@5, MRR, F1 temporal@5 y F1 por bins@5.
- **Descriptivo en desarrollo (no decide nada):** en cuántos de los 5 folds la selección automática elige un subconjunto con U (`reports/grupos_elegidos_por_fold.json`), y la comparación pareada en CV con Wilcoxon bilateral.
- **Regla de decisión:** si H-U1 no se confirma, **U no entra al modelo desplegado** y se reporta como resultado negativo, aunque en desarrollo se vea una mejora. El modelo desplegado sigue siendo el de D-027 hasta entonces.
- **Limitaciones ya conocidas:** (1) el PCA de U se ajustó con los 72 videos de `train` de un split anterior (`reports/split.json` del 24-09 20:01) y no con el split de cada fold, así que esos videos también caen en folds de prueba de la CV: fuga **no supervisada**, sin etiquetas, que puede favorecer a U en desarrollo; los videos de confirmación **no** están en ese PCA. (2) El ASR sigue en `base` (D-033) y U depende sólo del audio, así que la confirmación no se mezcla con el cambio de ASR; tras el barrido con `small` se repite todo y se reporta aparte. (3) Hay 32 columnas de U frente a ~10 de audio: con ~120 videos el riesgo de sobreajuste es real, y es justo lo que esta prueba mide.
- **Copia de seguridad:** los reportes y modelos previos a esta corrida quedaron en `reports_iter2_base/` y `models_iter2_base/`.
- **Estado:** preregistrada (pendiente de confirmación).

### D-039 · 2026-09-24 · Resultado exploratorio del grupo U en desarrollo (121 videos): entra en la selección, pero no mejora
**Exploratorio, según D-038: no decide nada; la confirmación está pendiente de los videos nuevos.** CV por canal k=5, 121 videos, 101 canales, 97,5 h; `run_all.py --config configs/gpu.yaml --from experiments` (149 s).
- **¿Entra U en la selección automática?** Sí en 3 de 5 folds: `A+H+U` (2 folds) y `K+S+A+H+F+E+L+U` (1 fold); los otros 2 eligieron subconjuntos sin U. Elección de la base D-027 por fold: `K+S+A+H+F`, `A+E`, `A+E`, `K+S+A+H+F+E+L`, `A+E`.
- **¿Mejora?** No de forma demostrable. `lgbm_autogroups_U` frente a `lgbm_autogroups` (n = 117–119 videos, Wilcoxon bilateral):

  | Métrica | con U | sin U | p |
  |---|---|---|---|
  | NDCG@10 (primaria) | 0,622 | 0,626 | 0,94 |
  | P@5 | 0,103 | 0,089 | 0,25 |
  | MRR | 0,241 | 0,232 | 0,69 |
  | F1 temporal@5 | 0,081 | 0,085 | 0,87 |
  | F1 bins@5 | 0,095 | 0,083 | 0,24 |

  La métrica primaria no mejora (ligeramente peor); las secundarias apuntan a favor de U pero sin significancia, y con 5 comparaciones sin corrección. Con sólo el grupo U (`only_U`) el NDCG@10 es 0,564, peor que el procedimiento completo (p = 0,0125).
- **Contexto:** ambos procedimientos superan al azar en NDCG@10 (0,622 frente a 0,477, p < 0,0001). El grupo U no es lo que hace al modelo mejor que el azar.
- **Cautelas:** (1) el PCA de U tiene la fuga no supervisada de D-038, que favorecería a U; que aun así no mejore refuerza la lectura negativa. (2) Los números no son comparables con la iteración 2 (102 videos, NDCG 0,599): el dataset cambió y la base ahora elige `A+E` o `K+S+A+H+F` en vez de sólo audio, lo que muestra que la selección por fold es inestable. (3) Cada fold tiene sólo ~24 videos de test.
- **Efecto colateral:** la corrida regeneró `models/` con el procedimiento base D-027 (mayoría `A+E`), no con U. El modelo anterior está en `models_iter2_base/`. U **no** entró al modelo desplegado.
- **Estado:** vigente; H-U1 pendiente de confirmación en los videos del paquete incremental.

### D-040 · 2026-09-25 · Preregistro: entrenar con clips humanos frente a Most Replayed (paso 6.3)
**Escrito ANTES de correr `python -m src.evaluation.human_train` y sin haber visto ninguna métrica de este análisis.**
- **Pregunta (D-028):** si Most Replayed y la curación humana casi no correlacionan (Spearman −0,04 en 11 videos), ¿un modelo entrenado directamente con clips humanos los recupera mejor que uno entrenado con Most Replayed?
- **Datos:** **29 videos** de Repurpose-10K con `y_human` y features (26 canales, todos en inglés, 11 223 candidatos; tasa positiva media por video 0,356). Supera el umbral de D-031 (25) por poco. Los 32 «válidos» de `rebuild_status` incluyen 3 sin features. Los 11 videos de D-028 forman parte de estos 29.
- **Procedimiento (el del script, sin modificar):** validación cruzada por canal (k = 5, ~6 videos de prueba por fold), semilla 42, hiperparámetros sin tocar (D-020), los **7 grupos** de `configs/default.yaml` (sin U, para no mezclar con D-038 y sin selección automática de grupos). Dos modelos con las mismas features: `entrenado_con_humano` (relevancia graduada por solapamiento con clips humanos dentro de cada video) y `entrenado_con_mostreplayed`, más `aleatorio`. Métrica primaria: F1 temporal@5 contra los clips humanos; secundarias P@5, MRR y P@1.
- **Hipótesis (Wilcoxon pareado por video, unilateral, α = 0,05):**
  - **H-H1 (primaria):** el modelo entrenado con humano supera al entrenado con Most Replayed en F1 temporal@5 (`humano_vs_mostreplayed_tF1`).
  - **H-H2:** el modelo con humano supera al azar (`entrenado_con_humano_vs_aleatorio_tF1`).
  - **H-H3 (replica D-028, esperada no significativa):** el entrenado con Most Replayed no supera al azar.
- **Regla de decisión:** H-H1 se confirma sólo si p < 0,05. Si no, se reporta como **no confirmada**, aunque las medias apunten a favor del modelo humano. No se cambian modelo, features ni k tras ver el resultado.
- **Poder estadístico:** con 29 videos y ~6 por fold, un resultado no significativo **no** demuestra que las etiquetas sean equivalentes; sólo que esta muestra no lo detecta. Con la tasa positiva de 0,356, el azar ya acierta bastante, lo que reduce el margen. Hay más riesgo de sobreajuste: ~23 videos de entrenamiento por fold y hasta ~130 features.
- **Réplica:** esta corrida es el análisis primario para estos 29 videos y **se ejecuta una sola vez**. Los videos nuevos de Repurpose que llegan de la laptop (43 con clips humanos allá) servirán de réplica independiente; la corrida sobre el total se reportará aparte.
- **Estado:** preregistrada (pendiente de resultado).

### D-041 · 2026-09-25 · Resultado del paso 6.3 (D-040): entrenar con clips humanos NO recupera mejor los clips humanos en esta muestra
Corrida única sobre 29 videos, 26 canales, CV por canal k = 5 (`reports/entrenamiento_humano.json`), tal como se preregistró.

| Modelo | P@5 | MRR | P@1 | **F1 temporal@5** |
|---|---|---|---|---|
| entrenado con clips humanos | 0,345 | 0,483 | 0,310 | 0,117 |
| entrenado con Most Replayed | 0,331 | 0,440 | 0,276 | 0,124 |
| aleatorio | 0,345 | 0,519 | 0,276 | 0,136 |

| Hipótesis | p (Wilcoxon unilateral) | Resultado |
|---|---|---|
| H-H1 (primaria): humano > Most Replayed en F1 temporal | 0,61 | **no confirmada** |
| H-H2: humano > azar | 0,82 | **no confirmada** |
| H-H3: Most Replayed > azar (réplica de D-028) | 0,61 | no significativa, como se esperaba |

- **Lectura:** con esta muestra no se puede afirmar que la etiqueta humana permita recuperar los clips de los editores mejor que Most Replayed, ni mejor que el azar. Las medias del modelo humano son incluso algo peores que las del azar en F1 temporal y MRR, e iguales en P@5.
- **Lo que NO significa:** que ambas etiquetas sean equivalentes. Con 29 videos (~6 por fold), ~23 de entrenamiento y hasta ~130 features, el modelo tiene pocas posibilidades de aprender nada: es un resultado de **falta de poder**, no una demostración de ausencia de efecto. La tasa positiva de 0,356 hace además que el azar ya acierte bastante en P@5.
- **Coherencia con D-028:** el modelo entrenado con Most Replayed tampoco supera al azar frente a los clips humanos, como ya se había medido con 11 videos.
- **Qué falta:** réplica con los videos nuevos de Repurpose (43 con clips humanos en la laptop) y, sobre todo, más videos; la línea de ~8,4 k videos de Repurpose-10K sigue siendo la que puede dar poder estadístico. No se modificó ningún parámetro tras ver este resultado.
- **Estado:** vigente; H-H1 no confirmada, pendiente de réplica.

### D-042 · 2026-09-25 · Incremental de la laptop, detección de bloqueos y conjuntos congelados
- **Restauración:** paquete incremental de la laptop (1 250 archivos, 1,17 GB, sincronizado por Syncthing tras varias caídas por reposo de la laptop) restaurado con `unpack_transfer.py --no-repo`: sólo datos; el repo, `models/` y `reports/` de esta máquina siguen mandando. Se validaron 22 FLAC nuevos (175 audios) y se respaldaron las features previas.
- **Conjunto de desarrollo de D-038 congelado antes de mezclar nada:** `reports/conjunto_desarrollo_D038.json` (121 IDs). Todo video que llegue después es de **confirmación** de H-U1. El PCA de U se reajustó **sólo con el train** de `reports/split.json` (80 videos de desarrollo), sin ningún video nuevo.
- **Muestra humana de D-040 congelada:** `reports/conjunto_humano_D040.json` guarda los 29 IDs originales de D-040/D-041 y los 28 videos con `y_human` que quedan como **réplica independiente** (uno de ellos, `RBrt3D5rG9Q`, es de desarrollo para U, lo que no interfiere: son hipótesis distintas). Repetir el experimento sobre los 57 se reporta como «muestra ampliada», no como confirmación de D-040.
- **Corrección a D-039:** el video `RBrt3D5rG9Q` entró en D-039 con el grupo E incompleto (68 de 90 columnas); la laptop lo completó y ahora tiene las columnas `e_*` y `y_human`. Efecto esperado mínimo (1 de 121 videos); no se repitió D-039.
- **Adquisición (`acquire`):** detecta bloqueos anti-bot (mensajes «Sign in to confirm you're not a bot», HTTP 429, «too many requests»); tras 2 seguidos pausa 30 min (`ingest.block_max_consecutive`, `ingest.block_pause_s`) y cualquier resultado que no sea bloqueo reinicia la cuenta. Los videos fallidos se reintentan al volver a ejecutar. Pausa entre descargas de 4 a 6 s (con variación aleatoria). Cubierto por `tests/test_acquire_bloqueos.py`. **Límite conocido:** el HTTP 403 no cuenta como bloqueo (aparece de forma aislada en videos restringidos); 3 fallos 403 en ~50 descargas, ninguna racha.
- **`featurize_loop --una-vez`:** una sola pasada; sin ella espera 90 min por videos que nunca tendrán transcripción.
- **Primera tanda del 6.4:** 56 pendientes de `youtube_mr.csv` y 2 de Repurpose; `youtube_mr.csv` pasó de 92 a 139/152 válidos; total **196 videos válidos, 158,6 h, 157 canales** (108 es / 88 en); 0 bloqueos. ASR `base` en GPU float32, frente a `base` int8 en CPU de la laptop para el resto (~1 % de texto, D-036); el barrido único con `small` de D-033 lo iguala.
- **Estado:** vigente.

### D-043 · 2026-09-25 · Ampliación a 250 videos por idioma: mismas consultas, resultados más profundos
- **Decisión:** `configs/gpu.yaml` fija `dataset_youtube.per_query: 50` (antes 25) y mantiene las mismas consultas, `MAX_PER_CHANNEL = 2`, duración 15–75 min y sólo videos con curva Most Replayed. El manifest se **amplía** (no se reconstruye): las filas existentes no cambian.
- **Justificación:** con `per_query: 25` las consultas ya devuelven los mismos resultados. Profundizar es el cambio menos invasivo; añadir consultas nuevas cambiaría la composición temática y se decidiría aparte si esto no alcanza.
- **Uso de los videos nuevos:** todos son de **confirmación** (H-U1) o muestra ampliada; ninguno participa en decisiones de modelo ni de selección de features (D-021/D-038).
- **Riesgo:** cientos de peticiones a YouTube; se mantiene el ritmo de D-042 y la pausa de 30 min.
- **Estado:** vigente.

---
## Ingenieria inversa de la formula (2026-10-12)

### D-044 · 2026-10-12 · Exploratorio con 10 videos: que features acompanan al calor y metodo para que los clips caigan en los picos de retencion
**Exploratorio. No cambia el modelo desplegado ni decide nada de D-038/D-040.**
- **Datos:** 10 de los 29 videos de D-040 (todos en desarrollo D-038), 1 por canal, semilla 42, elegidos de forma determinista antes de ver resultados (`scripts/reverse_engineering_formula.py`). Todos en ingles. No se tocan los 28 de replica ni los de confirmacion de H-U1.
- **Procedimiento:** rho de Spearman PARCIAL (controlando posicion, rango y rango^2) entre cada feature (rango percentil dentro del video) y el calor `y_mr` / la superposicion con clips humanos; unidad = video; formula = mediana de rho de las features con |rho|>=0.08 y mismo signo en >=8/10 videos. Validacion leave-one-video-out (la formula del video evaluado se arma SIN ese video).
- **Resultado 1 (formula solo-contenido, LOVO):** Spearman con el calor 0.251 (posicion 0.185, modelo 0.215), NDCG@10 0.526 (modelo 0.496), pero F1 temporal@5 vs picos 0.082 frente a 0.056 del azar (p=0.63, 5/10 videos). Efecto pequeno, no distinguible del azar con n=10.
- **Resultado 2 (signos):** el calor se acompana de pausas (a_pause_ratio, l_silence, a_long_pauses_pm), variacion de tono (a_pitch_std), aplausos/vitores (e_applause_*, e_cheering_max), primera persona (s_first_person) y menos habla continua/palabras largas/sustantivos (a_speech_rate, l_word_len, k_noun_chunks_ps, e_speech_mean). La formula de la eleccion humana casi no existe (3 features, |rho|~0.10) y comparte 0 signos con la del calor: refuerza D-028.
- **Resultado 3 (metodo):** usar el calor como selector cuando el video de YouTube trae curva ("calor directo": ventana con mayor calor medio, NMS 0.3) da F1@5 0.511 y lift 3.33x frente a 0.102 / 1.30x del modelo (p=0.002, 10/10). Anadir calidad editorial (inicio no colgado y final completo) cuesta 0.04 de F1 (0.471) y 0.30 de lift (3.03x), a cambio de 0 % de inicios colgados (frente a 20 %). **Esto no es una prediccion**: usa la etiqueta como entrada, solo es posible con curva disponible.
- **Hallazgo adicional:** los clips humanos caen en zonas MENOS reproducidas que el promedio del video (lift 0.87x) y el calor directo no mejora la coincidencia con ellos (F1 vs humano 0.107, azar 0.157).
- **Limitaciones:** n=10, un solo idioma, contrastes sin correccion por comparaciones multiples, seleccion de umbrales (0.08, 8/10) fijada a priori pero sin sensibilidad. Cualquier uso confirmatorio requiere datos nuevos (D-021).
- **Estado:** vigente; resultados en `reports/reverse_engineering/`.

### D-045 · 2026-10-12 · Formula de contenido sobre los 121 videos de desarrollo: pausas y variacion acompanan al calor; resultado exploratorio
**Exploratorio (conjunto de desarrollo D-038; no hay confirmacion todavia). No cambia el modelo desplegado.**
- **Datos:** 121 videos / 101 canales (los de desarrollo; los 375 nuevos NO se usaron). 30 con clips humanos.
- **Regla fijada antes de correr** (docstring de `scripts/reverse_engineering_formula.py --dev121`): rho parcial (controlando posicion) promediado por canal; entra una feature si |rho mediano| >= 0.05 y prueba del signo con Bonferroni (0.05/68). Evaluacion dejando afuera el CANAL del video evaluado.
- **Formula (17 features):** + a_pause_ratio, l_silence, a_long_pauses_pm, a_rms_std, a_pitch_range, k_keybert_sim, a_pitch_std, e_music_max/mean, f_n_sentences, s_questions_pm, s_second_person; - a_voiced_ratio, e_speech_mean, k_noun_chunks_ps, a_speech_rate, a_rms_mean.
- **Resultados (n=121, Wilcoxon bilateral, sin correccion entre contrastes):** F1 temporal@5 vs picos 0.115 frente a azar 0.064 (p=0.0001), modelo LightGBM fuera de fold 0.093 (p=0.03); calor medio de los clips 1.32x el del video (azar 1.0x, modelo 1.27x); Spearman con el calor 0.204 (posicion 0.143, modelo 0.145). Con filtro de calidad editorial (inicio no colgado y final completo): F1 0.110, 0 % de inicios colgados (frente a 23 %).
- **Metodo con curva disponible:** calor directo F1 0.543 / 3.67x; calor + calidad F1 0.455 / 3.22x y 0 % inicios colgados. No es prediccion: usa la etiqueta como entrada.
- **Clips humanos:** ninguna feature supera el umbral; rho(calor, humano) = -0.035 (n=30). Ratifica D-028 con mas videos.
- **Limitaciones:** el efecto es pequeno en terminos absolutos (F1 0.11); la formula coincide en direccion con una parte de las features que el modelo ya usa, asi que la ventaja sobre LightGBM (p=0.03) es sugerente, no concluyente; todos los videos de humanos son en ingles; falta confirmar en videos nuevos (D-021) antes de adoptar.
- **Estado:** vigente; resultados en `reports/reverse_engineering_dev121/`.

### D-046 · 2026-10-12 · Preregistro: confirmacion de la formula de contenido (D-045) en los videos nuevos
**Escrito ANTES de correr `scripts/confirm_formula_D046.py` y sin haber evaluado ni una metrica de la formula en estos videos.**
- **Que se congela:** los 17 pesos de `reports/reverse_engineering_dev121/resumen.json` (clave `formula_calor`, ajustados SOLO con los 121 videos de desarrollo D-038). No se vuelve a ajustar ningun peso, umbral ni feature.
- **Datos (confirmacion):** todos los videos con features y curva Most Replayed que NO estan en el conjunto de desarrollo `conjunto_desarrollo_D038.json`, excluyendo los 28 de la replica humana de D-040 (para dejarla intacta). Ningun video de estos participo en D-021...D-045 en ninguna decision. Videos que pertenecen a un canal presente en desarrollo se mantienen, y se reporta ademas el analisis agregado por canal.
- **Procedimiento:** score = suma ponderada de los rangos percentiles (dentro del video) de las features; top-5 con NMS 0.3 (D-019); F1 temporal contra los bins con calor > 0.5 (max 20, Rhapsody); contraste pareado por video.
- **H-F1 (primaria, una sola):** la formula supera al azar en F1 temporal@5 (Wilcoxon pareado, unilateral, alfa = 0.05).
- **Secundarias (sin correccion, marcadas como tales):** formula vs modelo LightGBM desplegado (`models/ranker_lgbm.txt`, entrenado con 80 videos de desarrollo, por tanto fuera de muestra aqui) en F1 temporal@5 y calor medio de los clips / calor medio del video; variante formula + calidad editorial (inicio no colgado y final completo); desempeno por idioma.
- **Regla de decision:** si H-F1 no se confirma (p >= 0.05), la formula **no** se adopta y se reporta como resultado negativo, aunque en desarrollo se haya visto mejora. No se cambian pesos tras ver el resultado; una nueva version exigiria datos nuevos.
- **Confusores conocidos de antemano:** (1) las transcripciones de parte de estos videos son `base` en GPU float32 y las de desarrollo `base` int8 en CPU (~1 % de texto, D-036, D-042); (2) la curva Most Replayed mide re-reproduccion, no curacion editorial (D-028): confirmar esta formula NO dice nada sobre clips elegidos por humanos.
- **Estado:** preregistrada (pendiente de resultado). Se ejecuta una sola vez.

### D-047 · 2026-10-12 · Resultado de la confirmacion D-046: H-F1 CONFIRMADA; la formula de contenido supera al azar y al modelo desplegado en videos nuevos
Corrida unica (`scripts/confirm_formula_D046.py`), pesos congelados de D-045, **352 videos nuevos / 290 canales** (ES y EN), `reports/confirmacion_D046/`.

| Metodo | F1 temporal@5 vs picos | calor del clip / calor del video | inicio colgado |
|---|---|---|---|
| azar | 0.070 | 1.00x | – |
| modelo LightGBM desplegado (A+E) | 0.087 | – | – |
| **formula de contenido** | **0.131** | **1.27x** | 25 % |
| **formula + calidad editorial** | **0.126** | – | **0.3 %** |
| calor directo (usa la curva) | 0.542 | 3.49x | – |
| calor + calidad (usa la curva) | 0.469 | 3.07x | 0 % |

- **H-F1 (primaria): confirmada.** Formula > azar: dif +0.060, mejor en 215/352 videos, p = 4e-17 (unilateral). Por canal (290): +0.058, p = 5e-16.
- Secundarias (sin correccion): formula > modelo desplegado (+0.044, p = 1e-10); el modelo desplegado **no** supera al azar en estos videos (+0.016, p = 0.31). Por idioma, la formula supera al azar en es, en, es-US y en-US.
- **Chequeo posicional (descriptivo, fuera del preregistro):** un prior de posicion aprendido en desarrollo da F1 0.071 y la formula correlaciona 0.014 con la posicion, asi que la ganancia no es un efecto de posicion (formula > prior, p = 2e-12).
- **Lectura honesta:** el efecto es real pero moderado (F1 0.13, clips 1.27x mas "calientes" que el promedio). Predecir los picos solo con contenido sigue lejos de usar la curva (0.54). Nada de esto dice algo sobre clips elegidos por humanos (D-028).
- **Confusores:** ASR base GPU float32 vs int8 CPU en parte de los videos (~1 %, D-036/D-042); un solo pase, sin reajuste.
- **Estado:** vigente. La formula queda como candidata a reemplazar el ranking de contenido; adoptarla en el modelo desplegado requiere decision explicita del autor.

### D-048 · 2026-10-12 · Iteracion 1 en desarrollo y preregistro de V1 (formula con clips <= 45 s) para el criterio "atinar a la mayoria de videos"
**Parte exploratoria (121 videos de desarrollo, 101 canales, `scripts/iterar_formula_dev.py`, presupuesto declarado de 6 variantes, Bonferroni 0.05/6):** ninguna variante mejora el F1 temporal frente a la formula congelada V0 (D-045): V1 cortos -0.009, V2 diversificar -0.000, V3 contraste local -0.010, V4 todas las features +0.002, V5 ridge -0.010, V6 formula+contraste +0.002 (todas p > 0.4). Criterio "acierto" (hit@5 = al menos un clip del top-5 con `y_relevant` = 1): azar 23.9 %, V0 39.7 %, **V1 51.2 %** (McNemar exacto V1 vs V0: 17 vs 3, p = 0.0026). Control de artefacto: el azar restringido a clips <= 45 s sube solo a 27.7 %. Con la curva de YouTube: calor directo 96.7 %, calor + clips cortos 94.2 %.
**Preregistro (escrito ANTES de evaluar V1 en datos de confirmacion):**
- **V1:** misma formula congelada de D-045; solo compiten candidatos de duracion <= 45 s; top-5 con NMS 0.3. Sin ningun cambio de pesos ni umbrales.
- **Datos:** los mismos 352 videos de D-046 (no participaron en ninguna decision de V1; se declara que ya fueron usados una vez para V0, de modo que esta es una segunda hipotesis sobre los mismos videos).
- **H-V1 (primaria, una sola):** hit@5 de V1 > hit@5 de V0 (McNemar exacto, unilateral, alfa = 0.05).
- **Secundarias:** V1 vs azar restringido a <= 45 s en hit@5; F1 temporal y calor medio de V1 vs V0 (se espera que el F1 no suba).
- **Regla:** si H-V1 no se confirma, V1 no se adopta. Sin reajuste.
- **Estado:** preregistrada (pendiente de resultado).

### D-049 · 2026-10-12 · Resultado de D-048: H-V1 CONFIRMADA (hit@5 sube) con costo en F1; cierre del lazo de iteracion
Corrida unica sobre 352 videos / 290 canales (`reports/confirmacion_D048/`).

| Metodo | hit@5 (al menos un clip toca un pico) | F1 temporal@5 | calor clip / calor video |
|---|---|---|---|
| azar | 29.1 % | 0.070 | 1.00x |
| azar restringido a <= 45 s | 31.7 % | 0.060 | 1.00x |
| formula D-045 (V0) | 49.4 % | 0.131 | 1.27x |
| **V1 = formula + solo clips <= 45 s** | **53.4 %** | 0.118 | 1.29x |
| calor directo (usa la curva) | 95.5 % | 0.542 | 3.49x |

- **H-V1 confirmada:** V1 acierta en videos donde V0 no (30) mas que al reves (16); McNemar exacto unilateral p = 0.027. V1 vs azar corto: p = 2e-16.
- **Costo:** F1 baja -0.013 (p = 0.016): los clips cortos tocan un pico en mas videos pero lo cubren menos. Es un compromiso, no una mejora gratuita.
- **Lectura:** con solo contenido se acierta en ~53 % de los videos (azar ~32 %): una mayoria apenas. Para acertar en ~95 % hace falta la curva (calor directo). No hay una "formula secreta" de contenido que llegue mas alla.
- **Cierre del lazo:** 6 variantes en desarrollo sin mejora de F1 (D-048) y una confirmacion sobre los mismos 352 videos; seguir iterando sobre estos datos seria explorar sobre datos de confirmacion (D-021). Nuevas variantes requieren videos nuevos (D-043 sigue ampliando el dataset) o senales nuevas (visual, LLM).
- **Implementacion:** `ranking.method: auto` (`src/ranking/retention.py`); `ranking.max_clip_s: 45` activa V1 en el camino sin curva (semantica exacta de V1: sin filtro de calidad). La combinacion V1 + filtro de calidad NO esta confirmada. Por defecto sigue `lgbm`.
- **Estado:** vigente.

---
## Ampliacion y bateria de metodos (2026-10-12)

### D-050 · 2026-10-12 · Cohorte fresca, desarrollo ampliado y bateria de 9 modelos para la formula
**Escrito ANTES de ejecutar `scripts/bateria_metodos_D050.py` y antes de que existan features de la cohorte fresca.**
- **Cohorte fresca:** `scripts/ampliar_dataset_D050.py` anade al manifest videos de 32 consultas NUEVAS (otros generos y temas; `configs/ampliacion_D050.yaml`), meta 330 por idioma. Sus IDs quedan en `reports/conjunto_confirmacion_fresca_D050.json`. Se transcriben con `base` (D-033), igual que el resto. Ninguno participa en ninguna decision hasta la confirmacion.
- **Desarrollo ampliado:** los 352 videos de D-046/D-048, ya usados para confirmar V0 y V1, pasan a formar parte del conjunto de EXPLORACION junto con los 121 de desarrollo (473 videos). Consecuencia declarada: **dejan de servir como conjunto de confirmacion de H-U1**; H-U1 y cualquier metodo ganador se confirman en la cohorte fresca. Los 28 videos de la replica humana de D-040 siguen intactos.
- **Presupuesto de la bateria (fijado antes de correr):** 9 modelos (M1 LightGBM lambdarank, M2 regresion, M3 binario, M4 ridge, M5 logistica, M6 lambdarank con contexto local +-90 s, M7 ensamble formula+M1, M8 ensamble formula+M3, M9 binario con contexto), cada uno con 3 selecciones (todos, <= 45 s, <= 45 s + calidad). Hiperparametros fijos a priori (300 rondas, 15 hojas, min 200). Bonferroni sobre los 9 modelos (0.05/9). Validacion GroupKFold(10) por canal.
- **Criterio de ganador (para pasar a confirmacion):** supera a V1 en hit@5 (McNemar bilateral, p < 0.0056) **y** no pierde mas de 0.01 de F1 frente a V0 (Wilcoxon), o supera a V0 en F1 con p < 0.0056 sin perder hit@5. Si ninguno califica, la formula de contenido V0/V1 queda como la mejor conocida.
- **Confirmacion:** un solo pase del/los ganadores en la cohorte fresca, con pesos y modelos congelados antes de calcular features de esos videos.
- **Estado:** preregistrado (pendiente de resultado).

### D-051 · 2026-10-12 · Resultado de la bateria D-050: ningun modelo supera a la formula simple; el techo de contenido esta en ~50-55 % de acierto
Corrida unica sobre 473 videos / ~400 canales (121 desarrollo + 352 de D-046), GroupKFold(10) por canal, `reports/bateria_D050/`. Referencias: V0 (formula congelada, todos los candidatos): hit@5 47.7 %, F1 0.129; V1 (V0 con clips <= 45 s): hit@5 52.7 %, F1 0.115; azar 28.8 % / 0.070.

| Modelo (mejor seleccion) | hit@5 | F1 | lift | p hit vs V1 | p F1 vs V0 |
|---|---|---|---|---|---|
| M6 LightGBM lambdarank + contexto local (<= 45 s) | 54.7 % | 0.122 | 1.37 | 0.40 | 0.20 |
| M6 (todos los candidatos) | 51.6 % | 0.136 | 1.37 | 0.70 | 0.10 |
| M1 LightGBM lambdarank (todos) | 48.6 % | 0.135 | 1.36 | 0.085 | 0.16 |
| M2 regresion / M3 binario / M4 ridge / M5 logistica | 47-51 % | 0.10-0.13 | 1.31-1.36 | n.s. | n.s. o peor |
| M7/M8 ensambles formula + LightGBM | 48-51 % | 0.12-0.135 | 1.33-1.36 | n.s. | n.s. |
| M9 binario + contexto | 44-49 % | 0.09-0.11 | 1.26-1.31 | n.s. | peor |

- **Ningun modelo cumple el criterio de ganador de D-050** (p < 0.0056 en hit@5 contra V1, o en F1 contra V0). Las diferencias son de 1-3 puntos de hit@5 y 0.005-0.007 de F1, dentro del ruido.
- **Lectura:** modelos no lineales, contexto temporal y ensambles no extraen mas senal que 17 pesos lineales. El techo con features de audio y texto parece ser ~50-55 % de acierto y F1 ~0.13. Exigir <= 45 s ayuda al hit@5 (+4 a +5 puntos) a costa de F1 (-0.01 a -0.015) en todos los modelos; el filtro de calidad editorial cuesta F1 sin mejorar hit@5.
- **Consecuencia:** no hay candidato nuevo que confirmar. V0/V1 siguen siendo lo mejor conocido. La cohorte fresca (D-050) se usa para replicar V0/V1 congelados en generos nuevos (generalizacion), no para elegir.
- **Estado:** vigente.

### D-052 · 2026-10-12 · Preregistro: replicacion de V0 y V1 congeladas en la cohorte fresca (generos nuevos)
**Escrito ANTES de que existan features de la cohorte fresca (`conjunto_confirmacion_fresca_D050.json`).**
- **Congelado:** pesos de `models/formula_calor_D045.json`; V0 = todos los candidatos, V1 = candidatos <= 45 s; top-5 con NMS 0.3. Sin reajuste.
- **Datos:** videos de la cohorte fresca que tengan features y curva Most Replayed (esperados ~100, de 32 consultas nuevas). Si quedan < 40 validos, la replicacion se reporta como descriptiva.
- **H-R1 (primaria):** hit@5 de V1 > hit@5 del azar restringido a <= 45 s (Wilcoxon por video, unilateral, alfa = 0.05).
- **Secundarias:** F1 temporal@5 de V0 > azar (unilateral); hit@5 de V1 vs V0 (McNemar, bilateral); resultados por idioma; calor directo (techo).
- **Regla:** no se agregan metodos nuevos ni se cambian pesos tras ver el resultado. Es una replicacion, no una seleccion.
- **Estado:** preregistrada (pendiente de datos).

### D-053 · 2026-10-12 · Preregistro: piloto de senal visual (storyboards de YouTube) sobre los 121 videos de desarrollo
**Escrito ANTES de calcular ninguna feature visual ni mirar su relacion con el calor.**
- **Senal:** storyboard `sb1` de YouTube (frames de 160x90, ~1 cada 10 s). Por candidato: movimiento medio y maximo entre frames, tasa de cambios bruscos, desviacion del brillo, densidad de bordes y su contraste local (+-90 s). Gruesa a proposito (D-016 descartó video completo por costo).
- **Datos:** los 121 videos de desarrollo D-038 (el resto de la exploracion ya esta agotada). Ninguna decision con la cohorte fresca.
- **Presupuesto:** 3 modelos — (a) formula visual sola (pesos por rho parcial controlando posicion, dejando afuera el canal), (b) V0 + visual (lineal), (c) LightGBM lambdarank (M1) con y sin visual. GroupKFold por canal. Bonferroni 0.05/3.
- **Criterio para pasar a confirmacion fresca:** (b) supera a V0 o (c) supera a M1 en hit@5 (McNemar) o en F1 (Wilcoxon) con p < 0.0167. Si no, la senal visual gruesa se reporta como negativa.
- **Estado:** preregistrado (pendiente de resultado).


### D-054 Â· 2026-10-12 Â· Preregistro: modelo profundo de SECUENCIA temporal (video -> curva de calor) y fusion "modelo propone, curva valida"
**Escrito ANTES de entrenar cualquier red de secuencia.**
- **Motivacion:** hasta ahora cada candidato se puntuaba solo (D-051). Una red que ve la linea de tiempo completa puede usar la historia previa y posterior (acumulacion, pausa antes del pico).
- **Datos:** desarrollo ampliado (121 + 352 videos, tablas por bin: 100 bins por video, features de contenido K,S,A,H,F,E,L; sin posicion ni HuBERT). Cohorte fresca D-050 y replica humana intactas.
- **Control de posicion (D-018):** entrenamiento con recortes aleatorios de 40 bins e inferencia por ventanas deslizantes promediadas; el transformer no lleva codificacion posicional absoluta.
- **Presupuesto (12 configuraciones, 1 semilla, 10 folds por canal, 40 epocas fijas, sin early stopping):** MLP por bin (sin contexto); BiGRU 64 y 128; TCN 64 y 128; Transformer 64 y 128; perdidas MSE sobre rango z y ListNet; con y sin canal de contraste local. Bonferroni 0.05/12.
- **Linea base en los mismos bins y folds:** azar, formula V0 congelada, LightGBM lambdarank por bin, calor directo (techo).
- **Ganador (para confirmar en la cohorte fresca):** la configuracion de mayor F1 temporal@5 medio; se reentrena con todo el desarrollo ampliado y se evalua una sola vez en la cohorte fresca (Wilcoxon vs V0 y vs LightGBM por bin).
- **Fusion (exploratoria, mismos oof):** (A) calor, (B) modelo, (C) solo bins con calor en el 20 % superior, ordenados por modelo, (D) z(calor) + 0.5 z(modelo). Se reporta hit@5, F1 y acuerdo con los clips humanos (30 videos).
- **Estado:** preregistrado (pendiente de resultado).



### D-055 Â· 2026-10-12 Â· Resultado de D-054: la secuencia profunda no supera a LightGBM; la fusion "el modelo propone, la curva valida" llega a ~81-83 % de acierto
Corrida unica (`scripts/dl_secuencia_D054.py`), 444 videos del manifest previo a D-050 (los 57 de Repurpose viven en otro manifest y quedaron fuera del entrenamiento), 68 features, GroupKFold(10) por canal, `reports/dl_secuencia_D054/`.

| Metodo (bins de ~35 s) | hit@5 | F1 temporal@5 | calor clip / video |
|---|---|---|---|
| azar | 32.8 % | 0.055 | 1.01x |
| formula V0 | 53.4 % | 0.121 | 1.40x |
| **LightGBM lambdarank por bin** | **58.8 %** | **0.130** | **1.49x** |
| mejor red (BiGRU 128, MSE, dropout 0.3) | 54.7 % | 0.121 | 1.40x |
| BiGRU 64 + contraste local | 55.2 % | 0.117 | 1.46x |
| TCN / Transformer (mejores) | 53-55 % | 0.113-0.114 | 1.36x |
| calor directo (techo) | 98.6 % | 0.674 | 3.97x |

- **Ninguna de las 12 redes supera a V0** en F1 (p > 0.2 las mejores; las perdidas ListNet con TCN/Transformer quedan **peor**, p < 0.05) y varias quedan significativamente por debajo de LightGBM por bin (p < 0.03). El contexto temporal y la profundidad no extraen mas senal que el modelo de arboles con 68 features.
- **Fuera de muestra (29 videos Repurpose de D-040, nunca vistos en entrenamiento):** red BiGRU 62.1 % de acierto, F1 0.160 (azar 21.6 % / 0.046; formula V0 51.7 % / 0.138). Es n = 29, exploratorio.
- **Fusion "el modelo propone, la curva valida"** (mismos 29): validar con calor (20 % superior) y ordenar por modelo 82.8 % / F1 0.332; z(calor) + 0.5 z(modelo) 96.6 % / 0.643; calor directo 96.6 % / 0.690. En los oof de 444 videos: 80.9 % / 0.307 y 98.6 % / 0.642.
- **Frente a editores humanos todo queda en el nivel del azar** (F1 0.057-0.073 frente a 0.085 del azar): ni el calor, ni el modelo, ni la fusion recuperan lo que eligen editores (consistente con D-028). Ordenar por modelo dentro de los bins validados no mejora el acuerdo con editores (p = 0.26 frente a calor directo).
- **Estado:** vigente. El modelo final (3 semillas, todo el desarrollo ampliado) queda en `models/dl_secuencia_D054.pt`; su confirmacion en la cohorte fresca sigue pendiente de datos (D-052).



### D-056 Â· 2026-10-12 Â· Preregistro: el contenido SEMANTICO del habla (embeddings de texto por bin) y HuBERT, con LightGBM y BiGRU por bin
**Escrito ANTES de calcular los embeddings de texto por bin y sin haber visto ninguna metrica con ellos.**
- **Hipotesis:** las features manuales (K, S, H, F, L) capturan pocas propiedades del texto; un embedding semantico (sentence-transformers multilingue, el mismo de KeyBERT) del habla de cada bin puede aportar "de que se habla" y mejorar el acierto frente al modelo base.
- **Datos:** los mismos 444 videos y los mismos 10 folds por canal de D-054 (comparacion pareada con sus oof). Cohorte fresca y replica humana intactas.
- **Brazos (presupuesto 4 contrastes, Bonferroni 0.0125):** base (68 features) frente a base + texto (PCA 32 ajustado SOLO con el entrenamiento de cada fold), base + HuBERT (u_*, ya en las tablas; PCA con la fuga no supervisada de D-038), base + texto + HuBERT. Modelos: LightGBM lambdarank por bin y BiGRU 128 (MSE, dropout 0.3, recortes de 40 bins).
- **Criterio:** un brazo "gana" si supera al base del mismo modelo en F1 temporal@5 (Wilcoxon pareado, p < 0.0125) sin perder acierto; en ese caso se confirma una vez en la cohorte fresca. Si no, la senal semantica se reporta como negativa.
- **Estado:** preregistrado (pendiente de resultado).



### D-057 Â· 2026-10-12 Â· Resultado de D-053: la senal visual gruesa (storyboards) correlaciona con el calor pero no aporta sobre audio y texto
Corrida unica sobre 120 videos de desarrollo con storyboard `sb1` (frames de 160x90 cada ~10 s), GroupKFold(10) por canal, `reports/piloto_visual_D053/`.

| Modelo | hit@5 | F1 temporal@5 | calor clip / video |
|---|---|---|---|
| azar | 24.7 % | 0.065 | â€“ |
| solo visual | 36.7 % | 0.072 | 1.10x |
| V0 (formula) | 40.0 % | 0.124 | 1.35x |
| V0 + visual | 37.5 % | 0.125 | 1.38x |
| LightGBM (M1) | 40.8 % | 0.106 | 1.31x |
| M1 + visual | 38.3 % | 0.117 | 1.33x |

- **Criterio preregistrado (D-053): no se cumple.** V0+visual vs V0: acierto 8 videos ganados / 11 perdidos (p = 0.65), F1 +0.0004 (p = 0.53). M1+visual vs M1: F1 +0.011 (p = 0.40), acierto menor (p = 0.69). Ninguno alcanza p < 0.0167.
- **Hay senal, pero ya esta cubierta:** en cada fold entran 5 features visuales con rho parcial positivo y consistente entre canales (movimiento medio +0.08, tasa de cambios bruscos +0.06, variacion de brillo +0.08, contraste local de movimiento +0.04): los tramos con mas cambio visual tienden a ser mas re-reproducidos, pero esa informacion se solapa con la que ya aportan pausas, tono y volumen.
- **Artefacto a no citar:** en `resultado.json` la comparacion `vis_only_vs_azar` usa `hit > 0` sobre la probabilidad media del azar (0.247) y marca 96.7 % para el azar; el p = 4e-22 es un error de la comparacion, no un resultado. Los valores correctos son los de la tabla (36.7 % frente a 24.7 %).
- **Limitacion:** frames cada ~10 s a 160x90; no se mide cara, gestos ni texto en pantalla. No descarta que video completo con un modelo multimodal aporte.
- **Estado:** vigente. Resultado negativo segun el criterio.



### D-058 Â· 2026-10-12 Â· Resultado de D-056 (texto semantico y HuBERT) y preregistro de ingenieria de la etiqueta
**Resultado D-056** (444 videos, mismos 10 folds por canal que D-054, `reports/texto_semantico_D056/`):

| Modelo | Brazo | hit@5 | F1 | p F1 vs base | p hit vs base |
|---|---|---|---|---|---|
| LightGBM | base | 58.8 % | 0.130 | â€“ | â€“ |
| LightGBM | + texto (PCA 32) | 57.7 % | 0.129 | 0.95 | 0.60 |
| LightGBM | + HuBERT | 59.5 % | 0.136 | 0.109 | 0.80 |
| LightGBM | + texto + HuBERT | 58.6 % | 0.133 | 0.40 | 1.00 |
| BiGRU 128 | base | 54.7 % | 0.121 | â€“ | â€“ |
| BiGRU 128 | + texto | 52.7 % | 0.109 | 0.088 | 0.43 |
| BiGRU 128 | + HuBERT | 60.1 % | 0.134 | 0.028 | 0.028 |
| BiGRU 128 | + texto + HuBERT | 56.8 % | 0.122 | 0.79 | 0.45 |

- **Criterio D-056 (p < 0.0125): no se cumple.** El contenido semantico del habla (embeddings multilingues de cada bin) no mejora a ningun modelo; con la BiGRU incluso empeora. HuBERT muestra una tendencia pequena y consistente (+0.006 F1 con LightGBM, +0.013 con BiGRU, +5 puntos de acierto con BiGRU, p = 0.03) que **no sobrevive** a la correccion por 4 contrastes y arrastra la fuga no supervisada del PCA (D-038). Coherente con D-039.
- **Decision:** HuBERT se incluira como brazo SECUNDARIO exploratorio en la replicacion de la cohorte fresca (se calculan sus embeddings para esos videos); no se adopta salvo p < 0.05 alli, con el PCA ya congelado.

**Preregistro (escrito ANTES de entrenar con etiquetas alternativas): ingenieria de la etiqueta.**
- **Pregunta:** el modelo base predice el grado 0-4 por cuantiles del calor. Los "aciertos" son picos estrechos; quiza otra definicion de la etiqueta afine el orden en la cima.
- **Variantes (LightGBM por bin, 68 features, mismos 10 folds, 300 rondas, hiperparametros fijos; presupuesto 5 contrastes, Bonferroni 0.01):** (T1) binario sobre `y_relevant`; (T2) regresion sobre el calor z-escalado por video; (T3) regresion sobre la prominencia (calor menos mediana movil de +-5 bins) z-escalada; (T4) lambdarank con 10 grados por rango; (T5) regresion sobre log(calor). Referencia: lambdarank 5 grados (oof de D-054).
- **Criterio:** una variante supera a la referencia si mejora F1 temporal@5 con p < 0.01 sin perder acierto; en tal caso se confirma una vez en la cohorte fresca. Si no, el techo de contenido se mantiene.
- **Estado:** preregistrado (pendiente de resultado).



### D-059 Â· 2026-10-12 Â· Preregistro: representacion de las features (valores absolutos, contexto de video, vecinos)
**Escrito ANTES de entrenar con estas representaciones.**
- **Observacion:** todos los modelos ven solo el rango percentil de cada feature DENTRO del video. Se pierde (a) el nivel absoluto (un bin con volumen alto en un video callado), (b) el tipo de video (podcast, stand-up, conferencia), que cambia lo que hace "bueno" un momento, y (c) el entorno inmediato.
- **Brazos (LightGBM lambdarank por bin, 5 grados, mismos 444 videos y 10 folds por canal de D-054, 300 rondas, hiperparametros fijos; presupuesto 4 contrastes, Bonferroni 0.0125):** (R1) rango + valor absoluto; (R2) rango + media y desviacion por video de cada feature (contexto global, 136 columnas constantes por video); (R3) rango + rango de los bins vecinos (+-1 y +-2); (R4) todo lo anterior.
- **Control de posicion (D-018):** ninguna columna codifica la posicion (los vecinos se rellenan replicando el borde; las agregaciones por video no dependen del orden). Se verifica `p_rel_pos` fuera.
- **Criterio:** un brazo supera a la referencia (oof LightGBM base de D-054) si mejora F1 temporal@5 con p < 0.0125 sin perder acierto; en ese caso se confirma una vez en la cohorte fresca.
- **Estado:** preregistrado (pendiente de resultado).



### D-060 Â· 2026-10-12 Â· Resultado de D-058: ninguna etiqueta alternativa mejora el modelo por bin
Corrida unica, LightGBM por bin, 444 videos, 10 folds por canal, `reports/etiqueta_D058/`. Referencia: lambdarank con 5 grados (oof de D-054).

| Etiqueta | hit@5 | F1 | calor clip / video | p F1 | p hit |
|---|---|---|---|---|---|
| T0 lambdarank 5 grados (ref) | 58.8 % | 0.130 | 1.49x | â€“ | â€“ |
| T1 binario sobre `y_relevant` | 57.0 % | 0.126 | 1.45x | 0.57 | 0.42 |
| T2 regresion sobre calor z | 57.4 % | 0.130 | 1.49x | 0.76 | 0.51 |
| T3 regresion sobre prominencia | 56.8 % | 0.128 | 1.45x | 0.60 | 0.32 |
| T4 lambdarank 10 grados | 55.9 % | 0.127 | 1.47x | 0.60 | 0.07 |
| T5 regresion sobre log(calor) | 56.8 % | 0.131 | 1.50x | 0.99 | 0.35 |

- **Criterio D-058 (p < 0.01): no se cumple.** Cambiar lo que se le pide predecir no mueve el techo; las diferencias son de 1-3 puntos de acierto y <= 0.004 de F1 y todas van en contra o empatan.
- **Familias de informacion o de objetivo cerradas hasta ahora (todas negativas):** features de contenido adicionales (D-051), secuencia profunda (D-055), LLM juez (D-053), visual grueso (D-057), semantica del habla y HuBERT (D-058), etiqueta (esta).
- **Estado:** vigente.



### D-061 Â· 2026-10-12 Â· Resultado de D-059 (representacion de features) y preregistro de ajuste fino + confirmacion fresca de R4
**Resultado D-059** (444 videos, mismos 10 folds por canal, LightGBM por bin, `reports/representacion_D059/`):

| Brazo | hit@5 | F1 | calor clip / video | p F1 vs ref | p hit |
|---|---|---|---|---|---|
| R0 rango (ref) | 58.8 % | 0.130 | 1.49x | â€“ | â€“ |
| R1 rango + valor absoluto | 59.5 % | 0.139 | 1.54x | 0.053 | 0.80 |
| R2 rango + contexto del video (media, desviacion) | 59.0 % | 0.135 | 1.51x | 0.13 | 1.00 |
| R3 rango + bins vecinos (+-1, +-2) | 57.2 % | 0.134 | 1.51x | 0.26 | 0.46 |
| **R4 todo junto (476 columnas)** | **60.6 %** | **0.142** | **1.57x** | **0.0133** | 0.38 |

- **Criterio D-059 (p < 0.0125): no se cumple por muy poco** (Wilcoxon p = 0.0133; t pareada p = 0.019). Diferencia media de F1 de +0.0126, IC95 bootstrap [+0.002, +0.023]. Es la unica familia con direccion consistente en los cuatro brazos (todos mejoran el F1 frente a la referencia). Lectura: el nivel absoluto y el tipo de video aportan algo pequeno que el rango relativo descarta. No se declara mejora.
- **No contamina posicion:** ninguna columna `p_*`; vecinos con relleno de borde; agregados por video no dependen del orden.

**Preregistro (escrito ANTES de ajustar R4 y antes de mirar features de la cohorte fresca con R4):**
- **A. Ajuste fino sobre R4 (LightGBM por bin, mismos folds, presupuesto de 10 contrastes, Bonferroni 0.005 frente a R0):** (H1) 7 hojas, 400 rondas; (H2) 31 hojas; (H3) min_data_in_leaf 150; (H4) feature_fraction 0.3; (H5) feature_fraction 0.5, 500 rondas; (H6) lr 0.03, 600 rondas; (H7) lambda_l2 50, 400 rondas; (H8) extra_trees; (S5) promedio de 5 semillas de R4 con la configuracion original; (U) R4 + HuBERT (u_*).
- **B. Confirmacion fresca (una sola vez, pesos congelados, entrenado con el desarrollo ampliado; la configuracion de R4 original):** H-R4: R4 supera a R0 en F1 temporal@5 en la cohorte fresca (Wilcoxon pareado, unilateral, alfa = 0.05). Secundarias: acierto; BiGRU de D-054; HuBERT. Si la configuracion de A cumple el criterio de A, se anade como brazo secundario, no sustituye a H-R4.
- **Regla:** ninguna configuracion se adopta en el pipeline sin pasar B. No se reajusta tras ver la cohorte fresca.
- **Estado:** preregistrado (pendiente de resultado).



### D-062 Â· 2026-10-12 Â· Resultado de D-061 A: la representacion R4 mejora de forma robusta a la referencia; HuBERT suma el maximo; pasa a confirmacion fresca
Corrida unica sobre 444 videos, 10 folds por canal, LightGBM por bin (`reports/r4_ajuste_D061/`). Referencia R0 = solo rangos. Bonferroni sobre 10 contrastes (p < 0.005).

| Brazo | hit@5 | F1 | calor clip / video | p F1 vs R0 | p F1 vs R4 default |
|---|---|---|---|---|---|
| R0 rango (ref) | 58.8 % | 0.130 | 1.49x | â€“ | â€“ |
| R4 default | 60.6 % | 0.142 | 1.57x | 0.0133 | â€“ |
| H1 hojas 7 | 59.9 % | 0.140 | 1.54x | 0.028 | 0.61 |
| H2 hojas 31 | 59.7 % | 0.135 | 1.52x | 0.36 | 0.10 |
| H3 min_data 150 | 59.7 % | 0.140 | 1.53x | 0.031 | 0.40 |
| **H4 feature_fraction 0.3** | 60.8 % | **0.145** | 1.56x | **0.0025** | 0.63 |
| H5 ff 0.5, 500 rondas | 62.2 % | 0.141 | 1.56x | 0.023 | 0.42 |
| H6 lr 0.03, 600 rondas | 59.7 % | 0.139 | 1.54x | 0.087 | 0.39 |
| H7 lambda_l2 50 | 59.0 % | 0.140 | 1.54x | 0.060 | 0.43 |
| **H8 extra_trees** | 61.7 % | **0.145** | 1.57x | **0.0018** | 0.41 |
| S5 promedio de 5 semillas | 61.5 % | 0.142 | 1.56x | 0.020 | 0.29 |
| **U = R4 + HuBERT** | **61.7 %** | **0.147** | **1.59x** | **0.0002** | 0.10 |

- **Cumplen el criterio de A (p < 0.005 frente a R0): H4, H8 y U.** Ninguno es distinguible de R4 default (p >= 0.10): el ajuste fino no encuentra una configuracion claramente mejor, lo que sugiere que la ganancia viene de la REPRESENTACION y no de los hiperparametros.
- **Magnitud:** de 0.130 a 0.142-0.147 de F1 (+9 % a +13 % relativo) y de 58.8 % a 60.6-62.2 % de acierto. Es una mejora real y pequena, en el mismo orden de magnitud que la variabilidad entre folds.
- **Advertencia sobre HuBERT (U):** el PCA de `u_*` se ajusto con los 80 videos de entrenamiento del split de D-042, que tambien caen en folds de prueba de esta CV (fuga NO supervisada, D-038). Puede sobreestimar U; la confirmacion fresca la corrige porque usa el PCA congelado sobre videos que nunca vio.
- **Plan (D-061 B, sin cambios):** una sola corrida en la cohorte fresca, entrenando con el desarrollo ampliado: H-R4 (R4 > R0, primaria); secundarias marcadas: H4, H8 y U (cada una frente a R0). Los embeddings HuBERT de la cohorte fresca se calculan con el PCA congelado (`models/speech_pca.npz`), sin reajustarlo.
- **Estado:** vigente.



### D-063 Â· 2026-10-12 Â· Ensambles exploratorios y regla fija para la confirmacion fresca de la familia R4
**Exploratorio. Los 4 ensambles se anunciaron en la conversacion antes de calcularse pero no se registraron aqui antes de correrlos; se reportan como tales.** Promedio de z-scores por video de predicciones fuera de fold ya calculadas (sin reentrenar), `scripts/ensambles_D063.py`, 444 videos.

| Ensamble | hit@5 | F1 | calor clip / video | p F1 vs U | p hit vs U |
|---|---|---|---|---|---|
| U = LightGBM R4 + HuBERT (ref) | 61.7 % | 0.147 | 1.59x | â€“ | â€“ |
| E1 U + BiGRU (base + HuBERT) | 64.2 % | 0.153 | 1.61x | 0.38 | 0.21 |
| E2 R4 + H4 + H8 | 62.2 % | 0.146 | 1.58x | 0.54 | 0.87 |
| **E3 U + H8 + BiGRU (base + HuBERT)** | **65.1 %** | **0.156** | **1.64x** | 0.086 | 0.040 |
| E4 U + V0 | 59.9 % | 0.141 | 1.56x | 0.069 | 0.23 |

- **Ninguno supera a U de forma significativa** (umbral 0.0125). E3 mejora el acierto en 3.4 puntos (p = 0.04, sin correccion) y el F1 en +0.009 (p = 0.086). La tendencia es consistente con que modelos de familias distintas (arboles con R4, arboles con HuBERT, red de secuencia) se complementan, pero con n = 444 no se distingue del ruido.
- **Acumulado frente al modelo base de D-054 (LightGBM por bin con rangos):** de 58.8 % / 0.130 a 65.1 % / 0.156 (+6.3 puntos y +0.026 de F1, +20 % relativo). Cada escalon por separado es pequeno; el acumulado es el mayor movimiento observado en toda la busqueda, y esta sujeto a dos sesgos declarados: PCA de HuBERT con fuga no supervisada (D-038) y seleccion de configuraciones sobre los mismos folds (D-062).
- **Regla fija para la confirmacion fresca (escrita ANTES de calcular nada en la cohorte fresca):** una sola corrida con modelos entrenados en el desarrollo ampliado (444 videos) y aplicados a la cohorte fresca: R0 (referencia), R4 default (H-R4, primaria, unilateral alfa = 0.05), y como secundarias marcadas (Bonferroni 0.0125): U = R4 + HuBERT, H4, H8 y E3 (promedio de z-scores de U, H8 y la BiGRU con HuBERT). Sin reajuste de ningun hiperparametro ni peso.
- **Decision de adopcion:** nada entra al pipeline sin pasar la confirmacion fresca; si E3 no supera a R4 default alli, se adopta el mas simple que cumpla.
- **Estado:** preregistrado para la confirmacion fresca (pendiente de datos).



### D-064 Â· 2026-10-12 Â· Confirmaciones unicas en la cohorte fresca D-050 (93 videos, 86 canales, 32 consultas nuevas, ES y EN)
Una sola corrida de cada prueba, modelos y pesos congelados, entrenados con el desarrollo ampliado (444 videos). De 101 videos de la cohorte, 93 quedaron validos (8 descartados por el filtro de idioma, igual que en desarrollo). HuBERT con el PCA congelado (sha256[:16] = 87a80b42f96a4565, verificado antes y despues). Resultados en `reports/replicacion_D052/`, `reports/replicacion_DL_D054/`, `reports/confirmacion_fresca_D063/`.

**1. Formula de contenido V0/V1 (D-052): CONFIRMADA en generos nuevos y en los dos idiomas.**

| Metodo | hit@5 | F1 temporal@5 | calor clip / video |
|---|---|---|---|
| azar | 26.9 % | 0.072 | 1.00x |
| azar restringido a <= 45 s | 30.6 % | 0.063 | 1.00x |
| V0 (formula) | 47.3 % | 0.149 | 1.35x |
| V1 (formula + clips <= 45 s) | 54.8 % | 0.141 | 1.43x |
| calor directo (usa la curva) | 94.6 % | 0.555 | 3.25x |

- **H-R1 (primaria D-052): confirmada** (V1 frente a azar restringido a <= 45 s, p = 4e-6). V0 supera al azar en F1 (p = 3e-9). V1 frente a V0 en acierto: 11 contra 4 videos, p = 0.12. Por idioma: EN 52.3 % (azar 22.9 %), ES 57.1 % (azar 30.4 %).

**2. Familia R4 (D-061 B / D-063): NO CONFIRMADA** (entrenado con 444 videos, aplicado a los 93):

| Metodo | hit@5 | F1 | calor clip / video | dif F1 vs R0 | p F1 (unilateral) |
|---|---|---|---|---|---|
| azar | 31.9 % | 0.057 | 1.00x | â€“ | â€“ |
| R0 (rangos) | 62.4 % | 0.142 | 1.52x | â€“ | â€“ |
| **R4 (primaria)** | 62.4 % | 0.157 | 1.51x | +0.015 | **0.12** |
| U = R4 + HuBERT | 65.6 % | 0.147 | 1.53x | +0.005 | 0.26 |
| H4 ff 0.3 | 65.6 % | 0.158 | 1.58x | +0.016 | 0.074 |
| H8 extra_trees | 64.5 % | 0.154 | 1.51x | +0.013 | 0.10 |
| E3 ensamble | 68.8 % | 0.166 | 1.57x | +0.024 | 0.025 |
| calor directo | 96.8 % | 0.684 | 3.71x | â€“ | â€“ |

- **H-R4 (primaria, alfa = 0.05) no se confirma (p = 0.12).** Ninguna secundaria cumple Bonferroni 0.0125 (la mejor, E3, p = 0.025; acierto E3 frente a R0: 14 contra 8 videos, p = 0.14). Por la regla fijada en D-061/D-063, **ninguna variante se adopta**.
- **Direccion consistente, potencia insuficiente:** las cinco diferencias son positivas (+0.005 a +0.024), del mismo orden que en desarrollo (+0.013 para R4), pero con 93 videos y varianza alta por video el contraste no alcanza. Lectura prudente: el efecto, si existe, es de unos +0.01-0.02 de F1; hacen falta del orden de 300+ videos nuevos para detectarlo con fiabilidad. (Descriptivo, no preregistrado: combinar la p bilateral de desarrollo, 0.013, con la de la cohorte fresca, 0.12 unilateral, por Stouffer da p ~ 0.005; no se usa para decidir porque el desarrollo uso configuraciones elegidas sobre los mismos folds.)
- **Nota sobre la referencia:** R0 en la cohorte fresca rinde mas que en desarrollo (62.4 % / 0.142 frente a 58.8 % / 0.130), probablemente por mezcla de generos mas favorable; las comparaciones son pareadas dentro de la cohorte.

**3. Modelo profundo de secuencia congelado, BiGRU 128 (D-054): no supera a los modelos de arboles.**

| Metodo | hit@5 | F1 | calor clip / video |
|---|---|---|---|
| azar | 31.9 % | 0.057 | 1.00x |
| formula V0 (en bins) | 59.1 % | 0.134 | 1.44x |
| LightGBM por bin | 62.4 % | 0.142 | 1.52x |
| BiGRU 128 | 53.8 % | 0.129 | 1.37x |
| **valida con el 20 % superior del calor y ordena por BiGRU** | **84.9 %** | **0.327** | **2.45x** |

- BiGRU supera al azar (p = 0.0001) pero no a V0 (p = 0.70) ni a LightGBM por bin (p = 0.15). La fusion "el modelo propone, la curva valida" replica en datos nuevos el 81-83 % de desarrollo (84.9 %).

**Balance de la busqueda (D-051 a D-064):** 7 familias de informacion o de objetivo probadas contra el mejor modelo de contenido (features adicionales, secuencia profunda, LLM juez, visual grueso, semantica del habla, etiqueta, representacion y ajuste). La unica con direccion consistente (R4 y su ensamble) no alcanza confirmacion con n = 93. Lo confirmado en datos nuevos es: (a) V0/V1 supera al azar en ES y EN (~47-55 % de acierto frente a 27-31 %); (b) la curva, usada como selector, acierta en ~95-97 %; (c) la fusion modelo + curva en ~85 %.
- **Estado:** vigente.



### D-065 Â· 2026-10-12 Â· Preregistro: segunda cohorte fresca para resolver la familia R4 / ensamble E3
**Escrito ANTES de descargar ningun video de la cohorte 2 y sin haber evaluado nada en ella.**
- **Motivo:** en D-064 R4 y E3 mejoran el F1 en la direccion esperada (+0.015 y +0.024) pero con n = 93 no alcanzan significancia (p = 0.12 y 0.025). La cohorte 1 ya se uso para esa evaluacion y no puede reutilizarse como confirmacion independiente.
- **Datos:** ~100 videos nuevos de 32 consultas distintas a las de D-050 (`configs/ampliacion_D065.yaml`), ES y EN, mismo filtro de idioma y de curva Most Replayed. IDs en `reports/conjunto_confirmacion_fresca2_D065.json`. No participan en ninguna decision previa.
- **Modelos congelados:** los entrenados con el desarrollo ampliado de 444 videos (R0, R4, E3 = z(U) + z(H8) + z(BiGRU base + HuBERT)) con la configuracion exacta de D-063. Sin reentrenar con la cohorte 1.
- **Hipotesis (una sola primaria):** H-E3: E3 supera a R0 en F1 temporal@5 en la cohorte 2 (Wilcoxon pareado, unilateral, alfa = 0.05). Secundarias marcadas: R4 vs R0; acierto de E3 vs R0 (McNemar); evidencia combinada cohortes 1 + 2 por Stouffer ponderado por n (declarada ahora, descriptiva).
- **Regla:** si H-E3 se confirma, E3 se adopta como modelo de contenido para el camino sin curva; si no, se queda R0/V1 y R4/E3 se reporta como efecto pequeno no confirmado. No se reajusta nada tras ver la cohorte 2.
- **Disparo:** cuando el proceso de ampliacion termine.
- **Estado:** preregistrado (pendiente de datos).



### D-065 (enmienda) Â· 2026-10-12 09:27 Â· Hora limite de la busqueda y regla de cierre
**Escrita ANTES de cualquier dato de la cohorte 2.**
- **Instruccion del autor:** la busqueda se detiene a las **11:00** del 2026-10-12.
- **Disparo de la prueba unica D-065 (cambia respecto al texto original):** al terminar el proceso de ampliacion, o a las **10:50**, lo que ocurra primero. Si a las 10:50 la adquisicion no termino, la prueba corre **una sola vez** con los videos de la cohorte 2 ya featurizados en ese momento (todos, sin seleccion), y se reporta el n real. Si n < 40, se reporta como descriptiva.
- **No cambian:** modelos congelados, hipotesis primaria (E3 > R0 en F1 temporal@5, unilateral, alfa = 0.05), secundarias y regla de adopcion de D-065.
- **Cierre a las 11:00:** reporte final con el estado de todo, decisiones y entregables; el lazo se detiene.
- **Estado:** vigente.



### D-066 Â· 2026-10-12 Â· Ingenieria inversa exploratoria de los picos (444 videos; sin decision de modelo asociada)
**Exploratorio.** `scripts/ingenieria_inversa_D066.py` y `scripts/capitulos_D067.py`, `reports/ingenieria_inversa_D066/`.

1. **Que rodea a un pico (promedio disparado, 1 091 picos = maximos locales entre los 3 mayores de cada video, rango percentil - 0.5):** el efecto esta **solo en el bin del pico** (offset 0), no antes ni despues. Mas silencio (l_silence +0.121, z = +13.8), mas pausas (+0.109, z = +12.0) y pausas largas (+0.084), mas rango y variacion de tono (+0.087, +0.084), mas variacion de volumen (+0.087); menos habla continua (e_speech -0.081, velocidad -0.072, proporcion de voz -0.071) y menos volumen medio (-0.060). Coherente con que ni el contexto temporal (R3) ni las redes de secuencia (D-055) aportaran.
2. **Capitulos de YouTube:** 52 % de los videos los traen. El bin con un inicio de capitulo tiene rango de calor 0.620 frente a 0.489 en el resto (p = 4e-30), y el top-5 de calor contiene un inicio de capitulo en 65.5 % de los videos (azar 47.7 %). Es decir, parte de los picos son **puntos de navegacion** (la gente salta o vuelve al inicio de una seccion), no solo "momentos interesantes".
3. **El patron de pausas NO es un artefacto de capitulos:** rho parcial de l_silence con el calor +0.142 con capitulos y +0.182 sin ellos; controlando el inicio de capitulo sigue en +0.118. Los picos tienden a ser **fronteras entre segmentos** (silencio, cambio de tono, fin de una idea), con o sin capitulos declarados.
4. **Los capitulos no mejoran al modelo (pese al efecto univariado):** LightGBM por bin, rangos frente a rangos + capitulos: F1 0.130 -> 0.136 en los 444 (p = 0.35); con capitulos (n = 226) 0.120 -> 0.130 (p = 0.22); sin capitulos 0.140 -> 0.142. La informacion de capitulos se solapa con el silencio y la prosodia que el modelo ya ve.
5. **Cuando falla el modelo:** el F1 cae en las curvas mas "picudas" (un solo pico dominante): cuartiles de picudez Q1-Q4 = 0.139 / 0.150 / 0.141 / **0.089**. Duracion (rho +0.04) y popularidad (log vistas, -0.03) no influyen. Los metadatos (duracion, vistas, idioma, capitulos) como contexto no mejoran: F1 0.130 -> 0.128 (p = 0.76).
6. **El "acierto" bruto depende de cuantos bins son destacados**, no solo del modelo (rho +0.46 con el numero de bins destacados). Frente al azar de ESE video, el modelo suma **+26.7 puntos (IC95 +-4.1)** de acierto, con mejora en todos los cuartiles (+16, +33, +37, +23 puntos). Se recomienda reportar el exceso sobre el azar por video y no el acierto bruto.
- **Estado:** vigente.



### D-068 Â· 2026-10-12 Â· Posicion y vocabulario de los picos (exploratorio) y preregistro de una prueba de lexico aprendido dentro de cada fold
**Resultado exploratorio** (`scripts/picos_extremos_D068.py`, 444 videos; los z no estan corregidos por las ~miles de palabras evaluadas, asi que son orientativos):
- **Posicion:** el maximo global cae en los primeros 5 bins en 2.5 % de los videos (azar 5 %) y en los ultimos 5 en 5.2 %; en las curvas picudas se desplaza hacia adelante (mediana bin 59 frente a 40 en las planas). Los picos **no** son un efecto de "inicio del video".
- **Palabras sobre-representadas en los bins de pico:** ES "mira" (8.8 % vs 5.0 %, z = +3.5), "dios", "tienes", "historia", "madre", "miedo", "perdon", "hermano"; EN "let's", "called", "thank", "number", "episode", "name", "first", "who". **Sub-representadas:** conectores de continuidad ES "entonces" (18.6 % vs 25.6 %, z = -4.6), "sea" (-4.4), "pero", "para", "ser"; EN "then", "going", "was", "could", "before", "things".
- **Lectura:** coherente con picos = fronteras entre segmentos (aperturas, presentaciones, cierres de una anecdota o pasajes personales/emocionales) y menos discurso continuo de explicacion. Hipotesis, no resultado confirmado.

**Preregistro (antes de entrenar):** Â¿un lexico aprendido SOLO con el entrenamiento de cada fold mejora a LightGBM por bin?
- **Construccion por fold:** palabras (>= 3 letras) con >= 15 apariciones en bins de pico (top-3 de calor por video) del entrenamiento y |z| > 2.0 frente al resto, por idioma; pesos = log-odds; tres features por bin: media de pesos de las palabras presentes, suma de pesos positivos, suma de pesos negativos.
- **Contraste unico:** R0 + lexico frente a R0 (oof de D-054, mismos folds), F1 temporal@5, Wilcoxon bilateral, alfa = 0.05. Exploratorio; si p < 0.05 pasa como brazo secundario a la cohorte 2 (D-065).
- **Estado:** preregistrado (pendiente de resultado).



### D-068 (resultado) · 2026-10-12 · Lexico aprendido dentro de cada fold: NEGATIVO
R0 + lexico: F1 temporal@5 0.130 -> 0.128, Wilcoxon p = 0.88; aciertos solo-lexico 19 frente a solo-R0 29. No pasa a la cohorte 2. El vocabulario de los picos es descriptivo pero no aporta senal predictiva adicional a las features de R0.

### D-069 · 2026-10-12 · Rendimiento y firma por categoria de YouTube (exploratorio)
`scripts/generos_D069.py`, 444 videos. Exceso de hit sobre el azar: Entertainment (n=105) 0.312; People & Blogs (99) 0.325; Education (84) 0.176; Comedy (56) 0.262; News (25) 0.361; Sports (19) 0.265. Kruskal-Wallis del exceso entre categorias p = 0.241 (sin diferencia demostrable). La firma de silencio SI varia por categoria (p < 0.001): Comedy y Sports muestran asociacion mas fuerte con silencio/tono/risa-aplausos (rho ~0.26-0.30). Education es la categoria con menor F1 (0.108).

### D-070 · 2026-10-12 · Categoria como contexto del modelo (exploratorio): NEGATIVO
`scripts/categoria_D070.py`: R0 + one-hot de categoria, mismos folds. F1 0.130 -> 0.131 (dif +0.0013, Wilcoxon p = 0.45); hit 58.8 % -> 56.5 %. Sin mejora; el modelo no necesita el genero como contexto.

### D-071 · 2026-10-12 · Ampliacion de entrenamiento en cola y demo con un video fijo (sin decision de modelo)
- `scripts/ampliar_dataset_D071.py` + `configs/ampliacion_D071.yaml`: ~200 videos nuevos (24 consultas nuevas) SOLO para entrenamiento; arranca al terminar D-065; ids en `reports/conjunto_entrenamiento_extra_D071.json`.
- Demo del video del PPT (`-Gzz2LZGowU`, `scripts/demo_video_ppt.py`, `demo_probar_metodos_ppt.py`): con solo contenido (V0, LightGBM fuera de fold) 0/5 clips tocan el pico (52:48-53:24, pico unico y angosto); fusion (top 10 % calor) 1/5 en el puesto 3; calor directo 3/5 con el #1 sobre el pico. Un video no es evidencia de generalizacion; se reporta como ilustracion y caso dificil.
- Interfaz: `scripts/interfaz_clips.py` (URL -> clips) y pagina `entregables/demo_opusclip/index.html`.
- D-065: confirmacion de cohorte 2 lanzada a las 10:50 (HuBERT -> `confirmacion_cohorte2_D065.py`); pendiente de resultado en `C:\tesis_data\logs\confirmacion_c2_D065.log`.

### D-073 · 2026-10-12 · Submodelo de las 17 features de la formula + vecinos, y mezcla con R0 (exploratorio, desarrollo)
`scripts/subconjunto17_D072x.py`, 444 videos, mismos folds por canal que D-054. Sin decision de modelo; una de muchas pruebas del dia, por lo que el p no esta corregido.
- LightGBM solo con las 17 features de V0 + rango de la formula + vecinos +-1, +-2: F1 0.130 -> 0.135 (p = 0.26, n.s.).
- Mismo modelo promediado en 5 semillas y mezclado por rangos con R0: hit 58.8 % -> 59.5 %, **F1 0.130 -> 0.140 (+0.010, p = 0.015)**. Direccion igual que R4/E3 (D-064): ganancias de ~0.01-0.02 que necesitan >300 videos nuevos para confirmarse.
- **Estado:** hipotesis para la cohorte de D-071 (preregistrar antes de mirarla). No se adopta.

### D-072 · 2026-10-12 · Resultado D-065: cohorte fresca 2 (n = 63): la familia R4 y el ensamble E3 NO se confirman; R0/V1 siguen vigentes
Corrida unica lanzada a las 10:50 (HuBERT con PCA congelado sha256[:16] = 87a80b42f96a4565 -> `confirmacion_cohorte2_D065.py`), entrenado con 444 videos de desarrollo. Cohorte 2 usable con HuBERT: **63 de 99** (los que quedaron featurizados; la muestra se definio por disponibilidad, sin mirar resultados). Log: `C:\tesis_data\logs\confirmacion_c2_D065.log`.

| Metodo | hit@5 | F1 | lift |
|---|---|---|---|
| azar | 30.0 % | 0.051 | 0.99x |
| R0 (referencia) | 49.2 % | 0.108 | 1.37x |
| R4 | 47.6 % | 0.094 | 1.29x |
| U (R4 + HuBERT) | 49.2 % | 0.110 | 1.33x |
| H4 | 46.0 % | 0.099 | 1.33x |
| H8 | 44.4 % | 0.104 | 1.31x |
| E3 | 47.6 % | 0.109 | 1.34x |
| calor directo | 98.4 % | 0.658 | 4.12x |

- **H-E3 no se confirma:** dif F1 +0.0016, p = 0.58 (unilateral); acierto solo-E3 6 vs solo-R0 7. R4 (primaria): dif -0.0133, p = 0.86. Ninguna variante supera a R0.
- **Evidencia combinada con la cohorte 1 (Stouffer, descriptiva):** R4 p = 0.42; E3 p = 0.083. La ventaja aparente de E3 en la cohorte 1 (p = 0.025) no se sostiene en datos nuevos.
- **Regla de adopcion (D-061/D-063): no se adopta ninguna variante.** Siguen vigentes V0/V1 (formula) y R0 como referencia de aprendizaje.
- **Lectura:** las ganancias de R4/E3 observadas en desarrollo y en la cohorte 1 eran de ~+0.01-0.02 y no se replican; compatibles con ruido. Todo metodo de solo contenido sigue en ~45-50 % de acierto en esta cohorte (azar 30 %).
- **Consecuencia para D-073:** la mezcla R0 + submodelo de 17 features (exploratoria, +0.010 F1 en desarrollo) debe leerse con la misma cautela.