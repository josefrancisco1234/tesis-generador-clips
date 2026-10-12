"""Mini pagina web local (solo libreria estandar): pegas el link de YouTube -> clips 9:16, por que se eligieron
y validacion contra el mapa de calor real.

Uso:  python scripts/web_clips.py        y abrir http://localhost:8765
El sistema elige SOLO con el contenido; la curva de YouTube se usa despues, unicamente para calificar.
"""
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUTPUTS = Path(r"C:\tesis_data\outputs")
HEATMAPS = Path(r"D:\tesis_data\data\interim\heatmaps")
PY = r"C:\tesis_data\.venv\Scripts\python.exe"
PORT = 8765
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glosario_variables as GL
YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})")
JOBS: dict[str, dict] = {}
EXAMPLES = [
    {"id": "l_-igAkbVII", "lang": "ES", "note": "ejemplo 1 (4/5)"},
    {"id": "UIC10AA34YU", "lang": "EN", "note": "ejemplo 2 (3/5)"},
    {"id": "3KfZ30O2fEQ", "lang": "EN", "note": "ejemplo 3 (5/5 en evaluacion)"},
]


def latest_dir(vid: str):
    best = None
    for d in OUTPUTS.glob(f"{vid}_*"):
        if (d / "clips.json").exists() and (best is None or (d / "clips.json").stat().st_mtime > (best / "clips.json").stat().st_mtime):
            best = d
    return best


def get_heatmap(vid: str):
    p = HEATMAPS / f"{vid}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("heatmap")
    try:   # video fuera del dataset: intento pedirla a YouTube (puede fallar por anti-bot; entonces no hay validacion)
        from src.ingest import youtube
        meta = youtube.fetch_metadata(vid)
        hm = (meta or {}).get("heatmap")
        if hm:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return hm
    except Exception:  # noqa: BLE001
        return None


def build_result(d: Path) -> dict:
    j = json.loads((d / "clips.json").read_text(encoding="utf-8"))
    m = YT_ID.search(d.name.replace("_", "&v=", 1)) or re.match(r"([\w-]{11})_", d.name)
    vid = d.name[:11]
    res = {"dir": d.name, "vid": vid, "duration": j["duration_s"], "language": j["language"], "model": j["model"],
           "elapsed": j.get("elapsed_s"), "keywords": [k["term"] for k in j.get("video_keywords", [])][:10], "clips": [],
           "heat": None, "hot": [], "hits": None}
    hm = get_heatmap(vid)
    st = en = vals = None
    if hm:
        import numpy as np
        st = np.array([b["start_time"] for b in hm]); en = np.array([b["end_time"] for b in hm])
        vals = np.array([b["value"] for b in hm], float)
        if len(vals) > 1:
            vals[0] = vals[1]
        idx = np.nonzero(vals > 0.5)[0]
        idx = idx[np.argsort(vals[idx])[::-1]][:20]
        if len(idx) == 0:
            idx = np.argsort(vals)[::-1][:5]
        hot = [(float(st[i]), float(en[i])) for i in idx]
        res["heat"] = [{"t": float((a + b) / 2), "v": float(v)} for a, b, v in zip(st, en, vals)]
        res["hot"] = [{"a": a, "b": b} for a, b in hot]
        res["peak"] = {"a": float(st[int(vals.argmax())]), "b": float(en[int(vals.argmax())]), "v": float(vals.max())}
        res["mean_heat"] = float(vals.mean())
    hits = 0
    for c in j["clips"]:
        f = d / Path(c.get("render", {}).get("path", "")).name if c.get("render") else None
        item = {"rank": c["rank"], "start": c["start"], "end": c["end"], "score": c["score_final"],
                "hook": c["score_breakdown"]["hook"], "flow": c["score_breakdown"]["flow"], "value": c["score_breakdown"]["value"],
                "title": c["context"]["title"], "hookText": c["context"].get("hook"), "text": c["text"],
                "terms": c.get("highlighted_terms", [])[:8], "keywords": [k["term"] for k in c.get("keywords", [])][:6],
                "file": f.name if f and f.exists() else None, "explain": c.get("explain"), "dim_vars": c.get("dim_vars"), "touch": None, "heat": None}
        if vals is not None:
            msk = (en > c["start"]) & (st < c["end"])
            item["heat"] = float(vals[msk].mean()) if msk.any() else None
            item["touch"] = any(min(c["end"], b) - max(c["start"], a) > 0 for a, b in hot)
            hits += int(item["touch"])
        res["clips"].append(item)
    if vals is not None:
        res["hits"] = hits
    return res


def run_job(job_id: str, url: str, curva: bool):
    job = JOBS[job_id]
    cmd = [PY, "-u", str(ROOT / "scripts" / "generar_clips.py"), url] + (["curva"] if curva else [])
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace")
    for line in p.stdout:
        job["log"].append(line.rstrip())
        m = re.search(r"Clips en:\s*(.+)", line)
        if m:
            job["dir"] = Path(m.group(1).strip()).name
    job["code"] = p.wait()
    job["done"] = True


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self.send_json({"error": "no"}, 404)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        url = str(body.get("url", "")).strip()
        m = YT_ID.search(url)
        if not m:
            return self.send_json({"error": "Pega un link de YouTube valido"}, 400)
        vid = m.group(1)
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"log": [], "done": False, "dir": None, "code": None, "vid": vid, "t0": time.time()}
        threading.Thread(target=run_job, args=(jid, f"https://www.youtube.com/watch?v={vid}", bool(body.get("curva"))),
                         daemon=True).start()
        self.send_json({"job": jid})

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            b = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        if u.path == "/api/glossary":
            return self.send_json({"order": GL.ORDER, "vars": {k: {"group": v[0], "formula": v[1], "info": v[2], "loc": GL.LOC.get(k)} for k, v in GL.G.items()},
                                   "formulas": [{"name": a, "formula": b, "info": c} for a, b, c in GL.FORMULAS]})
        if u.path == "/api/examples":
            return self.send_json([{**e, "ready": bool(latest_dir(e["id"]))} for e in EXAMPLES])
        if u.path == "/api/status":
            j = JOBS.get(q.get("job", [""])[0])
            if not j:
                return self.send_json({"error": "job"}, 404)
            full = " | ".join(j["log"]).lower()
            stage = 0
            for k, pat in enumerate(("video descargado", "faster_whisper", "oraciones", "cargando modelo de embeddings",
                                     "metodo de seleccion", "clip exportado", "listo:"), 1):
                if re.search(pat, full):
                    stage = k
            if "se reutiliza" in full:
                stage = max(stage, 1)
            rendered = len(re.findall(r"clip exportado", full))
            return self.send_json({"done": j["done"], "code": j["code"], "dir": j["dir"], "log": j["log"][-12:],
                                   "secs": int(time.time() - j["t0"]), "stage": stage, "rendered": rendered, "vid": j["vid"]})
        if u.path == "/api/result":
            vid, dname = q.get("vid", [None])[0], q.get("dir", [None])[0]
            d = (OUTPUTS / dname) if dname else (latest_dir(vid) if vid else None)
            if not d or not (d / "clips.json").exists():
                return self.send_json({"error": "sin resultado"}, 404)
            return self.send_json(build_result(d))
        if u.path == "/media":
            d, f = q.get("d", [""])[0], q.get("f", [""])[0]
            p = (OUTPUTS / d / f).resolve()
            if OUTPUTS.resolve() not in p.parents or not p.is_file():
                return self.send_json({"error": "no"}, 404)
            size = p.stat().st_size
            rng = self.headers.get("Range")
            a, b = 0, size - 1
            code = 200
            if rng and (mm := re.match(r"bytes=(\d*)-(\d*)", rng)):
                a = int(mm.group(1) or 0)
                b = int(mm.group(2)) if mm.group(2) else size - 1
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(b - a + 1))
            if code == 206:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.end_headers()
            with open(p, "rb") as fh:
                fh.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = fh.read(min(1 << 16, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    left -= len(chunk)
            return
        self.send_json({"error": "no"}, 404)


PAGE = r"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generador de clips</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--tx:#e8ebf0;--mut:#8b93a3;--ok:#22c55e;--acc:#7c5cff;--hot:#ef4444}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.5 "Segoe UI",system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;padding:14px 24px;border-bottom:1px solid var(--line)}
.logo{width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#7c5cff,#22c55e)}header span{color:var(--mut)}
.wrap{max-width:1300px;margin:0 auto;padding:18px 24px}
.row{display:flex;gap:10px;flex-wrap:wrap}
input[type=text]{flex:1;min-width:280px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:11px 14px;color:var(--tx);font-size:14px}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:10px 18px;font-weight:600;cursor:pointer}
button.ghost{background:var(--card);border:1px solid var(--line);color:var(--tx);font-weight:500;padding:7px 12px;font-size:13px}
button:disabled{opacity:.5;cursor:default}
.ex{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap;align-items:center;color:var(--mut)}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:14px 0}
.mut{color:var(--mut)}.sub{color:var(--mut);font-size:12.5px}
pre{background:#0b0d11;border:1px solid var(--line);border-radius:10px;padding:10px;max-height:190px;overflow:auto;font-size:11.5px;color:#b7c0cc;margin:8px 0 0;white-space:pre-wrap}
.bar{height:6px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:8px}.bar i{display:block;height:100%;width:30%;background:var(--acc);animation:m 1.2s infinite}
@keyframes m{0%{margin-left:-30%}100%{margin-left:100%}}
.big{font-size:30px;font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}
.clip{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.clip video{width:100%;aspect-ratio:9/16;background:#000;display:block;object-fit:cover}
.pad{padding:12px 14px}.top{display:flex;justify-content:space-between;gap:10px}
.score{min-width:50px;height:50px;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:17px;border:3px solid}
.chip{display:inline-block;padding:3px 9px;border-radius:99px;font-size:12px;font-weight:600;margin:6px 0}
.y{background:#10341f;color:#7ff0a8}.n{background:#2a2a30;color:#aab}
.t{color:var(--mut);font-size:12.5px;max-height:4.4em;overflow:hidden}
details{margin-top:8px}summary{cursor:pointer;color:#b6a8ff;font-size:13px}
table{width:100%;border-collapse:collapse;font-size:11.5px;margin-top:6px}
th,td{border-bottom:1px solid var(--line);padding:3px 4px;text-align:left}th{color:var(--mut);font-weight:500}
.vi{color:#8b93a3;font-size:10.5px;line-height:1.25;font-weight:400}
.pos{color:#7ff0a8}.neg{color:#ff9a9a}
.cb{height:6px;background:var(--line);position:relative}.cb i{position:absolute;top:0;height:100%}
.mini{display:grid;grid-template-columns:42px 1fr 24px;gap:6px;font-size:11.5px;color:var(--mut);align-items:center;margin-top:3px}
.mini div{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.mini div i{display:block;height:100%;background:var(--acc)}
.verdict{margin-top:10px;padding:10px 12px;border-radius:10px;background:#0f2a1a;border:1px solid #1f6b3a;color:#b9f3cd}
.warn{background:#2a220f;border-color:#6b5a1f;color:#f3e2b9}
</style></head><body>
<header><div class="logo"></div><b>Generador de clips 9:16</b><span>· potencial de retención 0–99 · solo contenido · Tesis Núñez del Prado (U. de Lima)</span></header>
<div class="wrap">
 <div class="row"><input id="url" type="text" placeholder="Pega el link de YouTube…"><button id="go">Generar clips</button></div>
 <div class="ex"><span>Videos de ejemplo (cada clic corre la simulación completa):</span><span id="exs"></span></div>
 <div id="tabs" class="ex"></div>
 <div id="prog" class="panel" style="display:none"><b id="pst">Procesando…</b> <span class="sub" id="psec"></span><div class="bar"><i></i></div>
  <div id="steps" style="margin-top:10px;line-height:1.7"></div><pre id="plog"></pre></div>
 <div id="out"></div>
</div>
<script>
const $=id=>document.getElementById(id);
const fmt=s=>{s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return (h?h+":":"")+String(m).padStart(h?2:1,"0")+":"+String(x).padStart(2,"0")};
const col=s=>s>=85?"#22c55e":s>=60?"#eab308":"#9ca3af";
const vid=u=>{const m=u.match(/(?:v=|youtu\.be\/|shorts\/|live\/)([\w-]{11})/);return m?m[1]:null};
let poll=null;
const STEPS=["Descargar video","Transcribir (Whisper local)","Segmentar en candidatos","Calcular variables","Puntuar con la fórmula","Recortar 9:16 + subtítulos","Listo"];
const DONE=[];   // resultados de esta sesión: [{vid,dir,label}]
async function examples(){
  const ex=await (await fetch("/api/examples")).json();
  $("exs").innerHTML="";
  ex.forEach((e,i)=>{const b=document.createElement("button");b.className="ghost";
    b.textContent=`${e.lang} · video ${i+1} (${e.id})`;
    b.onclick=()=>{$("url").value="https://www.youtube.com/watch?v="+e.id;start();};
    $("exs").appendChild(b);});
}
function tabs(active){
  const t=$("tabs");t.innerHTML="";
  if(!DONE.length)return;
  t.insertAdjacentHTML("beforeend",'<span class="sub">Resultados de esta sesión:</span>');
  DONE.forEach(d=>{const b=document.createElement("button");b.className="ghost";
    b.style.cssText=d.dir===active?"border-color:#7c5cff;color:#fff":"";b.textContent="✓ "+d.label;
    b.onclick=()=>{show({dir:d.dir});};t.appendChild(b);});
}
function steps(stage,rendered){
  return STEPS.map((n,i)=>{const k=i+1;const st=stage>=k&&k<7?"✓":(stage+1===k?"…":"·");
    const ex=(k===6&&stage>=5&&stage<7)?` (${rendered}/5)`:"";
    const c=stage>=k?"#7ff0a8":(stage+1===k?"#fff":"#8b93a3");
    return `<div style="color:${c}">${stage>=k||k===7&&stage>=7?"✓":(stage+1===k?"⟳":"○")} ${n}${ex}</div>`;}).join("");
}
async function start(){
  const url=$("url").value.trim(); $("out").innerHTML="";
  const r=await fetch("/api/run",{method:"POST",body:JSON.stringify({url})}); const j=await r.json();
  if(j.error){alert(j.error);return}
  $("go").disabled=true;document.querySelectorAll("#exs button").forEach(b=>b.disabled=true);
  $("prog").style.display="block";$("pst").textContent="Simulación en curso: "+url;
  $("steps").innerHTML=steps(0,0);
  clearInterval(poll);
  poll=setInterval(async()=>{
    const s=await (await fetch("/api/status?job="+j.job)).json();
    $("plog").textContent=s.log.join(String.fromCharCode(10));$("plog").scrollTop=1e9;$("psec").textContent=Math.floor(s.secs/60)+" min "+(s.secs%60)+" s";
    $("steps").innerHTML=steps(s.stage,s.rendered);
    if(s.done){clearInterval(poll);$("go").disabled=false;document.querySelectorAll("#exs button").forEach(b=>b.disabled=false);
      if(s.code===0&&s.dir){$("pst").textContent="Listo";
        if(!DONE.find(d=>d.dir===s.dir))DONE.push({vid:s.vid,dir:s.dir,label:"video "+s.vid});
        $("steps").innerHTML=steps(7,5);show({dir:s.dir});}
      else{$("pst").textContent="Falló: revisa el registro (¿YouTube bloqueó la descarga?)";}
    }},2000);
}
async function show(q){
  const qs=q.dir?"dir="+encodeURIComponent(q.dir):"vid="+q.vid;
  const r=await fetch("/api/result?"+qs); if(!r.ok){$("out").innerHTML='<div class="panel">Sin resultado.</div>';return}
  const R=await r.json();$("prog").style.display="none";tabs(R.dir);render(R);
}
function chart(R){
  const cv=document.createElement("canvas");cv.style.width="100%";cv.height=210;
  setTimeout(()=>{
    const dpr=devicePixelRatio||1,W=cv.clientWidth,H=210;cv.width=W*dpr;cv.height=H*dpr;const x=cv.getContext("2d");x.scale(dpr,dpr);
    const D=R.duration,px=t=>42+(W-54)*t/D,py=v=>H-26-(H-48)*v;
    x.font="11px Segoe UI";x.fillStyle="#8b93a3";x.strokeStyle="#262b36";
    [0,.5,1].forEach(v=>{x.beginPath();x.moveTo(42,py(v));x.lineTo(W-10,py(v));x.setLineDash(v==.5?[3,3]:[]);x.stroke();x.fillText(v.toFixed(1),10,py(v)+4)});x.setLineDash([]);
    for(let m=0;m<=D/60;m+=10)x.fillText(m+" min",px(m*60)-14,H-8);
    if(R.heat){
      R.hot.forEach(h=>{x.fillStyle="rgba(239,68,68,.22)";x.fillRect(px(h.a),8,Math.max(px(h.b)-px(h.a),3),H-34)});
      x.beginPath();R.heat.forEach((p,i)=>i?x.lineTo(px(p.t),py(p.v)):x.moveTo(px(p.t),py(p.v)));
      x.lineTo(px(R.heat[R.heat.length-1].t),py(0));x.lineTo(px(R.heat[0].t),py(0));x.closePath();x.fillStyle="rgba(180,185,196,.33)";x.fill();
      x.beginPath();R.heat.forEach((p,i)=>i?x.lineTo(px(p.t),py(p.v)):x.moveTo(px(p.t),py(p.v)));x.strokeStyle="#9aa1ae";x.lineWidth=1.3;x.stroke();
    }
    R.clips.forEach(c=>{x.fillStyle=c.touch===false?"rgba(124,92,255,.55)":"rgba(34,197,94,.55)";
      const w=Math.max(px(c.end)-px(c.start),4);x.fillRect(px(c.start),8,w,H-34);
      x.fillStyle="#fff";x.font="bold 11px Segoe UI";x.fillText("#"+c.rank,px(c.start)+2,20)});
  },0);
  return cv;
}
function render(R){
  const o=$("out");o.innerHTML="";
  const hasHeat=R.heat!=null;
  let v=`<div class="panel"><div class="row" style="justify-content:space-between"><div><div class="big">${R.clips.length} clips</div>
  <div class="sub">${fmt(R.duration)} · idioma ${R.language} · método: ${R.model} · tardó ${R.elapsed?Math.round(R.elapsed/60*10)/10+" min":"—"} · <a style="color:#b6a8ff" href="https://youtu.be/${R.vid}" target="_blank">youtu.be/${R.vid}</a></div></div>`;
  if(hasHeat) v+=`<div style="text-align:right"><div class="big" style="color:${R.hits>=3?"#22c55e":"#eab308"}">${R.hits}/${R.clips.length}</div><div class="sub">clips que caen sobre un pico real</div></div>`;
  v+=`</div></div>`;
  o.insertAdjacentHTML("beforeend",v);
  const p=document.createElement("div");p.className="panel";
  p.innerHTML=`<b>Mapa de calor real de YouTube vs. clips elegidos</b><div class="sub">${hasHeat?"Gris = calor real (Most Replayed). Rojo = picos de retención (&gt; 0.5). Verde = clip que cae en un pico; morado = clip fuera de pico.":"No hay mapa de calor disponible para este video (YouTube no lo publica o está bloqueado): solo se muestran los clips."}</div>`;
  p.appendChild(chart(R));
  if(hasHeat){const d=document.createElement("div");d.className="verdict";
    d.innerHTML=`<b>Validación:</b> el sistema eligió los clips <i>sin ver</i> esta curva, solo con audio y texto. Después se compara: <b>${R.hits} de ${R.clips.length}</b> caen sobre un pico real (el azar acierta ~1.5 de 5). El pico más alto está en <b>${fmt(R.peak.a)}–${fmt(R.peak.b)}</b>.`;p.appendChild(d);}
  o.appendChild(p);
  const g=document.createElement("div");g.className="grid";
  R.clips.forEach(c=>{
    const e=document.createElement("div");e.className="clip";
    const ex=c.explain;
    let tb="";
    if(ex){
      const mx=Math.max(...ex.vars.map(r=>Math.abs(r.contrib)),1e-6);
      tb=`<details><summary><b>¿Por qué se eligió? Score global: 17 variables · puesto ${ex.rank} de ${ex.n_candidates} ventanas</b></summary>
      <div class="sub" style="margin-top:6px">raw = suma de (peso × puntaje/99 de cada variable). Verde suma al score, rojo resta.</div>
      <table><tr><th>variable (orden del código)</th><th>valor</th><th>puntaje 0–99</th><th>peso</th><th>aporte al score</th></tr>`+
      byCode(ex.vars).map(r=>`<tr><td><b>${r.var}</b>${vinfo(r.var)}</td><td>${r.value==null?"—":(+r.value).toPrecision(3)}</td><td>${Math.round(r.pct*99)}</td><td>${r.weight>0?"+":""}${r.weight.toFixed(2)}</td>
      <td class="${r.contrib>=0?"pos":"neg"}">${r.contrib>=0?"+":""}${r.contrib.toFixed(3)}</td></tr>`).join("")+`</table></details>`;
    }
    let dv="";
    if(c.dim_vars){
      const nm={hook:"Hook · gancho",flow:"Flow · fluidez",value:"Value · contenido, audio, emoción"};
      const sg=["hook","flow","value"].map(d=>{
        const rows=c.dim_vars[d]||[];const tot=rows.reduce((a,r)=>a+r.shap,0);const ab=rows.reduce((a,r)=>a+Math.abs(r.shap),0)||1;
        const tr=(r)=>`<tr><td><b>${r.var}</b>${vinfo(r.var)}</td><td>${r.value==null?"—":(+r.value).toPrecision(3)}</td><td>${Math.round((r.pct??0.5)*99)}</td>
          <td class="${r.shap>=0?"pos":"neg"}">${r.shap>=0?"+":""}${r.shap.toFixed(3)}</td><td>${Math.round(Math.abs(r.shap)/ab*100)}%</td></tr>`;
        return `<div style="margin-top:10px"><b>${nm[d]}</b> · puntaje de la dimensión <b>${c[d]}</b>/99 · suma de aportes ${tot>=0?"+":""}${tot.toFixed(2)}
        <div style="max-height:360px;overflow:auto"><table><tr><th>variable (en el orden del código)</th><th>valor</th><th>puntaje 0–99</th><th>aporte a ${d.toUpperCase()}</th><th>peso en ${d}</th></tr>`+
        byCode(rows).map(tr).join("")+`</table></div></div>`;}).join("");
      dv=`<details open><summary><b>Desglose por variable: cuánto sacó cada una y cuánto aportó a Hook, Flow y Value</b></summary>
      <div class="sub" style="margin-top:6px"><b>puntaje 0–99</b> = qué tan alta es esta variable en este clip frente a las demás ventanas del mismo video.
      <b>aporte</b> = cuánto sumó o restó a la dimensión (verde suma, rojo resta). Cómo se calcula cada variable: ver el <a href="#glosario" style="color:#b6a8ff">glosario al final de la página</a>.
      <b>Ojo:</b> estas tres dimensiones explican el clip; la selección y el score global salen de la fórmula de 17 variables (tabla de abajo).</div>${sg}</details>`;
    }
    e.innerHTML=(c.file?`<video src="/media?d=${encodeURIComponent(R.dir)}&f=${encodeURIComponent(c.file)}" controls preload="metadata"></video>`:`<div class="pad mut">sin mp4</div>`)+
    `<div class="pad"><div class="top"><div><div class="sub">Clip #${c.rank} · ${fmt(c.start)}–${fmt(c.end)} · ${Math.round(c.end-c.start)} s</div>
    <b>${c.title||""}</b></div><div class="score" style="color:${col(c.score)};border-color:${col(c.score)}">${c.score}</div></div>
    ${c.touch===null?"":`<span class="chip ${c.touch?"y":"n"}">${c.touch?"✓ cae en un pico de YouTube":"fuera del pico"} · calor ${c.heat==null?"—":c.heat.toFixed(2)}</span>`}
    <div class="t">“${(c.hookText||c.text).slice(0,200)}”</div>
    <div class="mini">Hook<div><i style="width:${c.hook}%"></i></div>${c.hook}</div>
    <div class="mini">Flow<div><i style="width:${c.flow}%"></i></div>${c.flow}</div>
    <div class="mini">Value<div><i style="width:${c.value}%"></i></div>${c.value}</div>
    ${tb}${dv}<details><summary>Texto completo y palabras clave</summary><div class="t" style="max-height:none">${c.text}</div>
    <div class="sub">Palabras clave: ${c.keywords.join(", ")}</div></details></div>`;
    g.appendChild(e);});
  o.appendChild(g);
  glosario(o);
}
let GLO=null;
const vinfo=k=>{const g=GLO&&GLO.vars[k];return g?`<div class="vi">${g.info}${g.loc?` · <code>${g.loc[0].split("/").pop()}:${g.loc[1]}</code>`:""}</div>`:""};
const ordIdx=k=>{const i=GLO&&GLO.order?GLO.order.indexOf(k):-1;return i<0?9999:i};
const byCode=rows=>rows.slice().sort((a,b)=>ordIdx(a.var)-ordIdx(b.var));
fetch("/api/glossary").then(r=>r.json()).then(g=>{GLO=g;});
async function glosario(o){
  if(!GLO)GLO=await (await fetch("/api/glossary")).json();
  const d=document.createElement("div");d.className="panel";d.id="glosario";
  const grp={};(GLO.order||Object.keys(GLO.vars)).forEach(k=>{const v=GLO.vars[k];if(v)(grp[v.group]=grp[v.group]||[]).push([k,v])});
  d.innerHTML=`<h3 style="margin:0 0 6px">Glosario: cómo se calcula cada fórmula y cada variable</h3>
  <div class="sub">Primero las fórmulas del sistema; después, las variables por grupo con su fórmula y qué significan.</div>
  <table style="margin-top:8px"><tr><th>fórmula</th><th>cómo se calcula</th><th>explicación</th></tr>`+
  GLO.formulas.map(f=>`<tr><td><b>${f.name}</b></td><td><code>${f.formula}</code></td><td>${f.info}</td></tr>`).join("")+`</table>`+
  Object.entries(grp).map(([g,arr])=>`<details style="margin-top:8px"><summary><b>${g}</b> · ${arr.length} variables</summary><table><tr><th>variable</th><th>fórmula</th><th>qué significa</th><th>en el código</th></tr>`+
    arr.map(([k,v])=>`<tr><td>${k}</td><td><code>${v.formula}</code></td><td>${v.info}</td><td>${v.loc?"<code>"+v.loc[0].split("/").pop()+":"+v.loc[1]+"</code>":""}</td></tr>`).join("")+`</table></details>`).join("");
  o.appendChild(d);
}
$("go").onclick=start;
$("url").addEventListener("keydown",e=>{if(e.key==="Enter")start()});
examples();
</script></body></html>"""

if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"Listo: http://localhost:{PORT}  (Ctrl+C para cerrar)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
