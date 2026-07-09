"""Local browser backend for streamed lidar_preprocessing visualization."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from wato_lidar_preprocessing.viz_data import ChunkVizData, load_chunk_viz_data

log = logging.getLogger(__name__)

_MAX_STATIC_WEB_PTS = 350_000
_MAX_DYNAMIC_WEB_PTS = 900_000


def _sample(xyz: np.ndarray, *fields: np.ndarray | None, max_points: int, seed: int):
    n = xyz.shape[0]
    if n <= max_points:
        idx = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, max_points, replace=False).astype(np.int64))
    return (xyz[idx],) + tuple(None if f is None else f[idx] for f in fields)


def _prepare_data(data: ChunkVizData) -> dict[str, np.ndarray | dict | bool]:
    static_xyz, static_intensity = _sample(
        data.static_xyz,
        data.static_intensity,
        max_points=_MAX_STATIC_WEB_PTS,
        seed=41,
    )
    (
        dynamic_xyz,
        dynamic_sweep_id,
        dynamic_intensity,
        dynamic_p_occ,
        dynamic_n_obs,
        dynamic_n_hits,
        dynamic_classification,
    ) = _sample(
        data.dynamic_xyz,
        data.dynamic_sweep_id,
        data.dynamic_intensity,
        data.dynamic_p_occ,
        data.dynamic_n_obs,
        data.dynamic_n_hits,
        data.dynamic_classification,
        max_points=_MAX_DYNAMIC_WEB_PTS,
        seed=43,
    )

    if dynamic_sweep_id is not None and dynamic_sweep_id.shape[0] > 0:
        order = np.argsort(dynamic_sweep_id, kind="stable")
        dynamic_xyz = dynamic_xyz[order]
        dynamic_sweep_id = dynamic_sweep_id[order]
        if dynamic_intensity is not None:
            dynamic_intensity = dynamic_intensity[order]
        if dynamic_p_occ is not None:
            dynamic_p_occ = dynamic_p_occ[order]
        if dynamic_n_obs is not None:
            dynamic_n_obs = dynamic_n_obs[order]
        if dynamic_n_hits is not None:
            dynamic_n_hits = dynamic_n_hits[order]
        if dynamic_classification is not None:
            dynamic_classification = dynamic_classification[order]

    bounds_src = [a for a in (static_xyz, dynamic_xyz) if a.shape[0] > 0]
    bounds = np.vstack(bounds_src) if bounds_src else np.zeros((1, 3), dtype=np.float32)
    sweeps = (
        np.unique(dynamic_sweep_id).astype(np.int32)
        if dynamic_sweep_id is not None and dynamic_sweep_id.shape[0] > 0
        else np.empty(0, dtype=np.int32)
    )
    return {
        "static_xyz": np.ascontiguousarray(static_xyz.astype(np.float32)),
        "dynamic_xyz": np.ascontiguousarray(dynamic_xyz.astype(np.float32)),
        "dynamic_sweep_id": np.ascontiguousarray(
            (dynamic_sweep_id if dynamic_sweep_id is not None else np.empty(0)).astype(
                np.int32
            )
        ),
        "static_intensity": np.ascontiguousarray(
            (
                static_intensity
                if static_intensity is not None
                else np.empty(static_xyz.shape[0], dtype=np.float32)
            ).astype(np.float32)
        ),
        "dynamic_intensity": np.ascontiguousarray(
            (
                dynamic_intensity
                if dynamic_intensity is not None
                else np.empty(dynamic_xyz.shape[0], dtype=np.float32)
            ).astype(np.float32)
        ),
        "dynamic_p_occ": np.ascontiguousarray(
            (
                dynamic_p_occ
                if dynamic_p_occ is not None
                else np.full(dynamic_xyz.shape[0], np.nan, dtype=np.float32)
            ).astype(np.float32)
        ),
        "dynamic_n_obs": np.ascontiguousarray(
            (
                dynamic_n_obs
                if dynamic_n_obs is not None
                else np.full(dynamic_xyz.shape[0], -1, dtype=np.int32)
            ).astype(np.int32)
        ),
        "dynamic_n_hits": np.ascontiguousarray(
            (
                dynamic_n_hits
                if dynamic_n_hits is not None
                else np.full(dynamic_xyz.shape[0], -1, dtype=np.int32)
            ).astype(np.int32)
        ),
        "dynamic_classification": np.ascontiguousarray(
            (
                dynamic_classification
                if dynamic_classification is not None
                else np.full(dynamic_xyz.shape[0], -1, dtype=np.int8)
            ).astype(np.int8)
        ),
        "meta": {
            "bag_id": data.bag_id,
            "chunk_id": data.chunk_id,
            "bounds_min": bounds.min(axis=0).astype(float).tolist(),
            "bounds_max": bounds.max(axis=0).astype(float).tolist(),
            "sweeps": sweeps.astype(int).tolist(),
            "has_static_intensity": static_intensity is not None,
            "has_dynamic_intensity": dynamic_intensity is not None,
            "has_voxel_diag": dynamic_p_occ is not None,
            "counts": {
                "static": int(static_xyz.shape[0]),
                "dynamic": int(dynamic_xyz.shape[0]),
                "static_total": int(data.static_xyz.shape[0]),
                "dynamic_total": int(data.dynamic_xyz.shape[0]),
            },
        },
    }


def serve_web_viz(bag_id: str, chunk_id: str, *, host: str, port: int) -> None:
    """Start a local HTTP server for a dynamic browser viewer."""
    prepared = _prepare_data(load_chunk_viz_data(bag_id, chunk_id))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            log.info("web viz: " + fmt, *args)

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", _render_web_html().encode())
                return
            if path == "/api/meta":
                self._send(
                    200,
                    "application/json",
                    json.dumps(prepared["meta"], separators=(",", ":")).encode(),
                )
                return
            key = path.removeprefix("/api/")
            if key in prepared and isinstance(prepared[key], np.ndarray):
                arr = prepared[key]
                self._send(200, "application/octet-stream", arr.tobytes())
                return
            self._send(404, "text/plain; charset=utf-8", b"not found")

    server = ThreadingHTTPServer((host, port), Handler)
    actual_host, actual_port = server.server_address
    log.info("web viewer listening at http://%s:%s", actual_host, actual_port)
    print(f"web viewer: http://{actual_host}:{actual_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("web viewer stopped")
    finally:
        server.server_close()


def _render_web_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WATO LiDAR Viz</title>
<style>
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #101113; color: #eef1f4; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
#gl { position: fixed; inset: 0; width: 100%; height: 100%; display: block; }
#panel { position: fixed; left: 16px; right: 16px; bottom: 16px; padding: 12px; background: rgba(18,20,23,.9); border: 1px solid rgba(255,255,255,.14); border-radius: 8px; display: grid; gap: 10px; }
.row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
#title { font-size: 14px; font-weight: 650; min-width: 260px; }
#meta { color: #b8c0cc; font-size: 12px; }
label { color: #dbe2ea; font-size: 12px; display: inline-flex; align-items: center; gap: 6px; }
button, select { height: 30px; color: #eef1f4; background: #22272d; border: 1px solid #3b4651; border-radius: 6px; padding: 0 10px; }
input[type="range"] { min-width: 180px; accent-color: #6fb7ff; }
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="panel">
  <div class="row"><div id="title">loading...</div><div id="meta"></div></div>
  <div class="row">
    <label>view <select id="view"><option value="top">top</option><option value="iso">iso</option><option value="side">side</option><option value="front">front</option></select></label>
    <label>mode <select id="mode"><option value="single">single sweep</option><option value="trail">trail</option><option value="all">all dynamic</option></select></label>
    <label>color <select id="color"><option value="uniform">static/dynamic</option><option value="sweep">sweep id</option><option value="height">height</option><option value="intensity">intensity</option><option value="p_occ">p_occ</option><option value="n_obs">n_obs</option><option value="n_hits">n_hits</option><option value="classification">classification</option></select></label>
    <label>sweep <input id="sweep" type="range" min="0" max="0" value="0"></label>
    <button id="prev">prev</button><button id="play">play</button><button id="next">next</button>
    <label><input id="static" type="checkbox" checked> static</label>
    <label><input id="dynamic" type="checkbox" checked> dynamic</label>
    <label>point <input id="point" type="range" min="1" max="8" value="2"></label>
  </div>
</div>
<script type="module">
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl', { antialias: true, preserveDrawingBuffer: true });
if (!gl) throw new Error('WebGL unavailable');
async function bin(path, Ctor) { const b = await (await fetch(path)).arrayBuffer(); return new Ctor(b); }
const meta = await (await fetch('/api/meta')).json();
document.getElementById('title').textContent = `${meta.bag_id} / ${meta.chunk_id}`;
const [staticXYZ, dynamicXYZ, dynamicSweep, dynI, dynPOcc, dynObs, dynHits, dynClass] = await Promise.all([
  bin('/api/static_xyz', Float32Array), bin('/api/dynamic_xyz', Float32Array),
  bin('/api/dynamic_sweep_id', Int32Array), bin('/api/dynamic_intensity', Float32Array),
  bin('/api/dynamic_p_occ', Float32Array), bin('/api/dynamic_n_obs', Int32Array),
  bin('/api/dynamic_n_hits', Int32Array), bin('/api/dynamic_classification', Int8Array)
]);
const sweepValues = meta.sweeps;
const sweepInput = document.getElementById('sweep');
sweepInput.max = Math.max(0, sweepValues.length - 1);
const colorInput = document.getElementById('color');
if (!meta.has_dynamic_intensity) [...colorInput.options].find(o => o.value === 'intensity').disabled = true;
if (!meta.has_voxel_diag) for (const v of ['p_occ','n_obs','n_hits','classification']) [...colorInput.options].find(o => o.value === v).disabled = true;
function shader(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, `attribute vec3 aPos; attribute vec3 aColor; uniform mat4 uMvp; uniform float uPointSize; varying vec3 vColor; void main(){ gl_Position=uMvp*vec4(aPos,1.0); gl_PointSize=uPointSize; vColor=aColor; }`));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `precision mediump float; varying vec3 vColor; void main(){ vec2 d=gl_PointCoord-vec2(.5); if(dot(d,d)>.25) discard; gl_FragColor=vec4(vColor,1.0); }`));
gl.linkProgram(program); gl.useProgram(program);
const locPos = gl.getAttribLocation(program, 'aPos'), locColor = gl.getAttribLocation(program, 'aColor');
const locMvp = gl.getUniformLocation(program, 'uMvp'), locPoint = gl.getUniformLocation(program, 'uPointSize');
function makeCloud(xyz, colors) { const pos=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,pos); gl.bufferData(gl.ARRAY_BUFFER,xyz,gl.STATIC_DRAW); const col=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,col); gl.bufferData(gl.ARRAY_BUFFER,colors,gl.STATIC_DRAW); return {pos,col,count:xyz.length/3}; }
function fillColor(n, rgb) { const out = new Float32Array(n*3); for(let i=0;i<n;i++) out.set(rgb,i*3); return out; }
const staticCloud = makeCloud(staticXYZ, fillColor(staticXYZ.length/3, [.29,.56,.85]));
let dynamicCloud = makeCloud(new Float32Array(0), new Float32Array(0));
function ramp(t){ t=Math.max(0,Math.min(1,t)); return [Math.max(0,Math.min(1,1.5-Math.abs(4*t-3))),Math.max(0,Math.min(1,1.5-Math.abs(4*t-2))),Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))]; }
function range(vals, idx, missing){ let lo=Infinity, hi=-Infinity; for(const i of idx){ const v=vals[i]; if(v!==missing && Number.isFinite(v)){ lo=Math.min(lo,v); hi=Math.max(hi,v); }} return [lo,hi]; }
function updateDynamic(){
  const mode=document.getElementById('mode').value, current=sweepValues[Number(sweepInput.value)] ?? 0, idx=[];
  for(let i=0;i<dynamicSweep.length;i++){ const sid=dynamicSweep[i]; if(mode==='all'||sid===current||(mode==='trail'&&sid<=current&&sid>current-5)) idx.push(i); }
  const xyz=new Float32Array(idx.length*3), col=new Float32Array(idx.length*3), cMode=colorInput.value;
  let zLo=Infinity,zHi=-Infinity; for(const i of idx){ const z=dynamicXYZ[i*3+2]; zLo=Math.min(zLo,z); zHi=Math.max(zHi,z); }
  const [iLo,iHi]=range(dynI,idx,NaN), [pLo,pHi]=range(dynPOcc,idx,NaN), [oLo,oHi]=range(dynObs,idx,-1), [hLo,hHi]=range(dynHits,idx,-1);
  const sMin=sweepValues[0]??0, sMax=sweepValues[sweepValues.length-1]??1;
  for(let j=0;j<idx.length;j++){ const i=idx[j]; xyz[j*3]=dynamicXYZ[i*3]; xyz[j*3+1]=dynamicXYZ[i*3+1]; xyz[j*3+2]=dynamicXYZ[i*3+2]; let c=[.91,.30,.24];
    if(cMode==='sweep') c=ramp((dynamicSweep[i]-sMin)/Math.max(1,sMax-sMin));
    else if(cMode==='height') c=ramp((dynamicXYZ[i*3+2]-zLo)/Math.max(.001,zHi-zLo));
    else if(cMode==='intensity') c=ramp((dynI[i]-iLo)/Math.max(.001,iHi-iLo));
    else if(cMode==='p_occ') c=ramp((dynPOcc[i]-pLo)/Math.max(.001,pHi-pLo));
    else if(cMode==='n_obs') c=ramp((dynObs[i]-oLo)/Math.max(1,oHi-oLo));
    else if(cMode==='n_hits') c=ramp((dynHits[i]-hLo)/Math.max(1,hHi-hLo));
    else if(cMode==='classification') c=[[.30,.56,.85],[.95,.85,.20],[.95,.55,.15],[.55,.55,.55],[.91,.30,.24]][dynClass[i]] || [.20,.20,.20];
    col.set(c,j*3);
  }
  dynamicCloud=makeCloud(xyz,col);
  document.getElementById('meta').textContent=`dynamic ${idx.length.toLocaleString()} / ${meta.counts.dynamic_total.toLocaleString()} · static ${meta.counts.static.toLocaleString()} / ${meta.counts.static_total.toLocaleString()}`;
}
function ortho(l,r,b,t,n,f){ return [2/(r-l),0,0,0,0,2/(t-b),0,0,0,0,-2/(f-n),0,-(r+l)/(r-l),-(t+b)/(t-b),-(f+n)/(f-n),1];}
function norm(v){const d=Math.hypot(v[0],v[1],v[2])||1;return[v[0]/d,v[1]/d,v[2]/d];} function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];} function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function lookAt(eye,center,up){const z=norm([eye[0]-center[0],eye[1]-center[1],eye[2]-center[2]]),x=norm(cross(up,z)),y=cross(z,x);return[x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1];}
function mul(a,b){const o=new Array(16).fill(0);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o;}
const mn=meta.bounds_min,mx=meta.bounds_max,center=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2],extent=Math.max(1,Math.hypot(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2]));
function mvp(){ const aspect=canvas.width/Math.max(1,canvas.height), half=extent*.58, dist=extent*1.8, view=document.getElementById('view').value; let eye=[center[0],center[1],center[2]+dist],up=[0,1,0]; if(view==='iso'){eye=[center[0]+dist,center[1]-dist,center[2]+dist];up=[0,0,1];} if(view==='side'){eye=[center[0],center[1]-dist,center[2]];up=[0,0,1];} if(view==='front'){eye=[center[0]-dist,center[1],center[2]];up=[0,0,1];} return mul(ortho(-half*aspect,half*aspect,-half,half,-dist*4,dist*4),lookAt(eye,center,up));}
function drawCloud(cloud){gl.bindBuffer(gl.ARRAY_BUFFER,cloud.pos);gl.vertexAttribPointer(locPos,3,gl.FLOAT,false,0,0);gl.enableVertexAttribArray(locPos);gl.bindBuffer(gl.ARRAY_BUFFER,cloud.col);gl.vertexAttribPointer(locColor,3,gl.FLOAT,false,0,0);gl.enableVertexAttribArray(locColor);gl.drawArrays(gl.POINTS,0,cloud.count);}
function render(){const dpr=window.devicePixelRatio||1,w=Math.floor(canvas.clientWidth*dpr),h=Math.floor(canvas.clientHeight*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h);}gl.clearColor(.06,.065,.075,1);gl.clear(gl.COLOR_BUFFER_BIT);gl.uniformMatrix4fv(locMvp,false,new Float32Array(mvp()));gl.uniform1f(locPoint,Number(document.getElementById('point').value)*dpr);if(document.getElementById('static').checked)drawCloud(staticCloud);if(document.getElementById('dynamic').checked)drawCloud(dynamicCloud);requestAnimationFrame(render);}
function refresh(){updateDynamic();}
for(const id of ['view','mode','color','sweep','static','dynamic','point']) document.getElementById(id).addEventListener('input',refresh);
document.getElementById('prev').onclick=()=>{sweepInput.value=Math.max(0,Number(sweepInput.value)-1);refresh();}; document.getElementById('next').onclick=()=>{sweepInput.value=Math.min(Number(sweepInput.max),Number(sweepInput.value)+1);refresh();};
let timer=null; document.getElementById('play').onclick=(ev)=>{ if(timer){clearInterval(timer);timer=null;ev.target.textContent='play';return;} ev.target.textContent='pause'; timer=setInterval(()=>{sweepInput.value=(Number(sweepInput.value)+1)%(Number(sweepInput.max)+1);refresh();},140); };
refresh(); render();
</script>
</body>
</html>"""
