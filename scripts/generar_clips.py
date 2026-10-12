"""
Script de un solo click: pega una URL de YouTube abajo, ejecuta este archivo,
y al terminar se abre sola la carpeta con los clips .mp4 listos para subir.

Uso:
    1. Cambia la variable URL de abajo por el video que quieras.
    2. Ejecuta (boton "Run" de VSCode, o en terminal):
         C:\\tesis_data\\.venv\\Scripts\\python.exe scripts\\generar_clips.py
       (o dale doble-click a generar_clips.bat, que trae el mismo comando)
    3. Espera (tarda varios minutos: descarga + transcribe + analiza + renderiza).
    4. Al terminar se abre sola la carpeta con los .mp4 listos.

Para el detalle de cada formula usada (KVSUM, z-score de audio, coherencia,
PANNs, entropia lexica, LightGBM LambdaRank) ver docs/codigo_explicado/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================================
# 1) PON AQUI LA URL DEL VIDEO QUE QUIERES CONVERTIR EN CLIPS
# ============================================================================
URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=-Gzz2LZGowU"  # la interfaz grafica pasa la URL por argumento

# Cuantos clips como maximo quieres al final (los de mejor score, tras NMS)
TOP_K = 5

# Perfil de configuracion: "configs/default.yaml" (CPU) o "configs/gpu.yaml"
# (si esta PC tiene GPU NVIDIA; es mucho mas rapido y usa Whisper 'small')
CONFIG = "configs/gpu.yaml"

# Calidad de descarga del video. 360p es buen balance velocidad/calidad para
# subir a TikTok/Reels. Si tu internet es lento, baja a "134" -> "133" (240p).
FORMATO_VIDEO = "134+bestaudio/134"
# ============================================================================


def main():
    # nos aseguramos de correr desde la raiz del proyecto (donde esta src/,
    # porque todo el codigo importa cosas como "from src.pipeline import run")
    raiz = Path(__file__).resolve().parent.parent
    os.chdir(raiz)
    sys.path.insert(0, str(raiz))

    from src.common.config import load_config
    from src.common.logging_utils import setup_logging
    from src.pipeline import run

    setup_logging()
    cfg = load_config(CONFIG)
    cfg["ingest"]["youtube_video_format"] = FORMATO_VIDEO  # mas liviano/rapido que el 1080p de fabrica
    # METODO DE SELECCION (D-045/D-047): "auto" usa la curva Most Replayed de YouTube si el video la
    # trae (calor + calidad: los clips caen sobre los picos de retencion) y, si no, la formula de
    # contenido congelada (pausas, variacion de tono, reaccion del publico...). "lgbm" = modelo anterior.
    cfg["ranking"]["max_clip_s"] = 45   # solo camino SIN curva: V1 (D-049) acierta un pico en ~53 % de videos vs ~49 %
    USAR_CURVA = len(sys.argv) > 2 and sys.argv[2] == "curva"   # por defecto SOLO contenido (como en un video no subido a YouTube)
    cfg["ranking"]["method"] = "auto" if USAR_CURVA else "formula"   # OJO: cfg.ranking devuelve una copia; hay que escribir en cfg["ranking"]

    outputs_dir = cfg.path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    antes = time.time()

    print(f"\nGenerando clips de: {URL}")
    print("Esto va a tardar varios minutos. Por dentro, run() hace, EN ORDEN:\n")
    print("  1) DESCARGAR el video (yt-dlp) si le pasamos una URL (no un archivo local).")
    print("  2) TRANSCRIBIR el audio con faster-whisper -> palabras con timestamp.")
    print("  3) SEGMENTAR en 'candidatos' de 15-90s alineados a oracion")
    print("     (ver docs/codigo_explicado/01_segmentar_candidatos.py).")
    print("  4) FEATURIZAR cada candidato: ~115 numeros por candidato (K/S/A/H/F/E/L).")
    print("     Cada grupo tiene su formula -> docs/codigo_explicado/02_featurizar_builder.py")
    print("       K keywords   : w_ij = log(tf_ij+1)*log(N/PF_i) / norma_coseno")
    print("       S sentimiento: polaridad = P(positivo) - P(negativo)")
    print("       A audio      : z_rms = (log(volumen) - promedio) / desviacion")
    print("       H hook       : 1 si las primeras palabras traen pregunta/numero/contraste")
    print("       F flow       : coherencia = similitud coseno entre oraciones consecutivas")
    print("       E eventos    : p(risa) = red neuronal PANNs CNN14 escuchando el audio")
    print("       L linguistico: H = -SUM[ p(palabra) * log2(p(palabra)) ]  (entropia lexica)")
    print("  5) PUNTUAR cada candidato (ranking.method = auto):")
    print("     a) si el video de YouTube trae la curva Most Replayed -> score = calor medio de la")
    print("        ventana (los clips caen sobre los picos de retencion);")
    print("     b) si no -> formula de contenido congelada (models/formula_calor_D045.json).")
    print("     En ambos casos solo compiten candidatos autocontenidos (inicio no colgado, final")
    print("     completo). El desglose Hook/Flow/Value sigue saliendo del modelo explicador (SHAP).")
    print("  6) NMS: se descartan candidatos que se solapan demasiado con uno mejor.")
    print("  7) CONTEXTO: titulo/hook por plantilla (o LLM si hay API key configurada).")
    print("  8) RENDER: reencuadre 9:16 (sigue al hablante) + subtitulos quemados (ffmpeg).\n")

    # run() con una URL descarga el video solo (src/pipeline.py -> resolve_input),
    # asi que NO hace falta llamar a yt-dlp a mano antes de esto.
    run(URL, cfg, top_k=TOP_K, render=True)

    # encontrar la carpeta de salida: la mas nueva creada/modificada durante esta
    # corrida (el nombre exacto de la carpeta depende de un hash interno del
    # video, asi que es mas simple/robusto buscarla por fecha que adivinarla)
    candidatas = [d for d in outputs_dir.iterdir() if d.is_dir() and d.stat().st_mtime >= antes - 2]
    out_dir = max(candidatas, key=lambda d: d.stat().st_mtime) if candidatas else outputs_dir

    print(f"\nListo. Clips en: {out_dir}")
    print("Dentro encontraras: clip_01_scoreXX.mp4 ... y clips.json con el detalle")
    print("completo de cada clip (score, desglose Hook/Flow/Value, keywords, texto).")

    # abrir la carpeta de resultados automaticamente (Windows)
    try:
        os.startfile(str(out_dir))  # type: ignore[attr-defined]
    except Exception:
        subprocess.run(["explorer", str(out_dir)])


if __name__ == "__main__":
    main()
