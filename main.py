"""
MAIN - Generador de clips 9:16 con puntaje de retencion (tesis J. F. Nunez del Prado, U. de Lima)

Ejecutar sin argumentos para ver el menu:      python main.py
O directo:
    python main.py web                          pagina web local (pegas el link y salen los clips)   http://localhost:8765
    python main.py formulas                     puntajes y formulas de los 3 videos de la demo (consola)
    python main.py clips <url>                  genera los clips de una URL de YouTube (carpeta se abre sola)
    python main.py interfaz                     ventana de escritorio (alternativa a la web)

El sistema elige los clips SOLO con el contenido (audio + texto). La curva "Most Replayed" de YouTube se usa despues,
unicamente para calificar.
"""
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VENV_PY = Path(r"C:\tesis_data\.venv\Scripts\python.exe")          # entorno con whisper, lightgbm, ffmpeg, etc.
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
EJEMPLOS = {
    "1": ("ES  4/5 - El caso Marina y Laura", "https://www.youtube.com/watch?v=l_-igAkbVII"),
    "2": ("EN  3/5 - Dr. Anita Philips", "https://www.youtube.com/watch?v=UIC10AA34YU"),
    "3": ("EN  3/5 - Men's Round Table", "https://www.youtube.com/watch?v=3KfZ30O2fEQ"),
}


def correr(*args: str) -> int:
    return subprocess.call([PY, *args], cwd=RAIZ)


def web() -> None:
    print("Abriendo http://localhost:8765 (Ctrl+C para cerrar)")
    webbrowser.open("http://localhost:8765")
    correr("scripts/web_clips.py")


def formulas() -> None:
    correr("docs/codigo_explicado/05_formulas_como_en_la_web.py")


def clips(url: str | None = None) -> None:
    if not url:
        print("\nVideos de ejemplo:")
        for k, (nombre, u) in EJEMPLOS.items():
            print(f"  {k}) {nombre}   {u}")
        eleccion = input("Numero de ejemplo o pega una URL: ").strip()
        url = EJEMPLOS[eleccion][1] if eleccion in EJEMPLOS else eleccion
    print(f"\nProcesando {url} (descarga, transcripcion, puntuacion y recorte 9:16; todo local) ...")
    correr("scripts/generar_clips.py", url)


def interfaz() -> None:
    correr("scripts/interfaz_clips.py")


MENU = {
    "1": ("Abrir la pagina web (pegar link -> clips + explicacion + mapa de calor)", web),
    "2": ("Ver puntajes y formulas de los 3 videos de ejemplo (consola)", formulas),
    "3": ("Generar los clips de una URL (mp4 9:16)", clips),
    "4": ("Abrir la ventana de escritorio", interfaz),
    "5": ("Abrir carpeta de documentos (formulas, mapa del codigo)", lambda: os.startfile(RAIZ / "docs")),
    "6": ("Abrir carpeta de resultados", lambda: os.startfile(RAIZ / "resultados")),
}


def main() -> None:
    a = sys.argv[1:]
    if a:
        if a[0] == "web":
            web()
        elif a[0] == "formulas":
            formulas()
        elif a[0] == "interfaz":
            interfaz()
        elif a[0] == "clips":
            clips(a[1] if len(a) > 1 else None)
        elif a[0].startswith("http"):
            clips(a[0])
        else:
            print(__doc__)
        return
    while True:
        print("\n=== Generador de clips 9:16 - menu ===")
        for k, (texto, _) in MENU.items():
            print(f"  {k}) {texto}")
        print("  0) Salir")
        op = input("Elige: ").strip()
        if op == "0":
            return
        if op in MENU:
            MENU[op][1]()


if __name__ == "__main__":
    main()
