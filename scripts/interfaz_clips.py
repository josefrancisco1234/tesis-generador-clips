"""Interfaz grafica: pegas el link de YouTube, das clic en "Generar clips" y se abre la carpeta con los mp4.

Ejecutar: doble clic en interfaz_clips.bat (o: python scripts/interfaz_clips.py).
Por dentro lanza scripts/generar_clips.py (mismo pipeline: transcribe, puntua, recorta 9:16 y quema subtitulos).
"""
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
YT = re.compile(r"(youtu\.be/|youtube\.com/(watch\?v=|shorts/|live/))[\w-]{11}")


EJEMPLOS = [
    "https://www.youtube.com/watch?v=l_-igAkbVII",   # ES - 4/5 clips tocan un pico
    "https://www.youtube.com/watch?v=UIC10AA34YU",   # EN - 4/5
    "https://www.youtube.com/watch?v=3KfZ30O2fEQ",   # EN - 5/5
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generador de clips 9:16 - Tesis Nunez del Prado")
        self.geometry("860x560")
        self.q: queue.Queue = queue.Queue()
        self.proc = None
        self.out_dir: Path | None = None

        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="Link del video de YouTube:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        fila = ttk.Frame(top)
        fila.pack(fill="x", pady=6)
        self.url = tk.StringVar(value=EJEMPLOS[0])
        ent = ttk.Combobox(fila, textvariable=self.url, values=EJEMPLOS, font=("Segoe UI", 11))
        ent.pack(side="left", fill="x", expand=True)
        ent.focus()
        self.btn = ttk.Button(fila, text="Generar clips", command=self.iniciar)
        self.btn.pack(side="left", padx=(8, 0))
        self.curva = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Usar la curva de calor de YouTube (solo demo; un video no subido NO la tiene)",
                        variable=self.curva).pack(anchor="w")
        self.btn_dir = ttk.Button(fila, text="Abrir carpeta", command=self.abrir, state="disabled")
        self.btn_dir.pack(side="left", padx=(8, 0))

        self.estado = tk.StringVar(value="Listo. Pega un link y presiona 'Generar clips'.")
        ttk.Label(top, textvariable=self.estado).pack(anchor="w")
        self.barra = ttk.Progressbar(top, mode="indeterminate")
        self.barra.pack(fill="x", pady=6)

        self.log = tk.Text(self, wrap="word", font=("Consolas", 9), bg="#101418", fg="#d7e0e8")
        sb = ttk.Scrollbar(self, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True, padx=(12, 0), pady=(0, 12))
        self.after(150, self.vaciar_cola)

    def iniciar(self):
        url = self.url.get().strip()
        if not YT.search(url):
            messagebox.showwarning("Link invalido", "Pega un link de YouTube (watch?v=..., youtu.be/... o shorts/...).")
            return
        self.btn.config(state="disabled")
        self.btn_dir.config(state="disabled")
        self.out_dir = None
        self.log.delete("1.0", "end")
        self.estado.set("Procesando... (descarga, transcripcion, puntuacion y render; tarda varios minutos)")
        self.barra.start(12)
        threading.Thread(target=self.correr, args=(url, self.curva.get()), daemon=True).start()

    def correr(self, url, curva=False):
        py = r"C:\tesis_data\.venv\Scripts\python.exe"
        if not os.path.exists(py):
            py = sys.executable
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self.proc = subprocess.Popen([py, "-u", str(RAIZ / "scripts" / "generar_clips.py"), url] + (["curva"] if curva else []), cwd=RAIZ, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace")
            for linea in self.proc.stdout:
                self.q.put(("log", linea))
            self.q.put(("fin", self.proc.wait()))
        except Exception as e:  # noqa: BLE001
            self.q.put(("log", f"ERROR: {e}\n"))
            self.q.put(("fin", 1))

    def vaciar_cola(self):
        try:
            while True:
                tipo, dato = self.q.get_nowait()
                if tipo == "log":
                    self.log.insert("end", dato)
                    self.log.see("end")
                    m = re.search(r"Clips en:\s*(.+)", dato)
                    if m:
                        self.out_dir = Path(m.group(1).strip())
                    elif "transcrib" in dato.lower():
                        self.estado.set("Transcribiendo el audio...")
                    elif "render" in dato.lower():
                        self.estado.set("Recortando a 9:16 y quemando subtitulos...")
                else:
                    self.barra.stop()
                    self.btn.config(state="normal")
                    if dato == 0:
                        self.estado.set(f"Listo. Clips en: {self.out_dir}" if self.out_dir else "Listo.")
                        if self.out_dir:
                            self.btn_dir.config(state="normal")
                    else:
                        self.estado.set("Fallo. Revisa el registro de abajo.")
        except queue.Empty:
            pass
        self.after(150, self.vaciar_cola)

    def abrir(self):
        if self.out_dir and self.out_dir.exists():
            os.startfile(str(self.out_dir))  # type: ignore[attr-defined]


if __name__ == "__main__":
    App().mainloop()
