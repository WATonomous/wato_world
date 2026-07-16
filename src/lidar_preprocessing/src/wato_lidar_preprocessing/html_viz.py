"""Self-contained browser viewer for lidar_preprocessing point clouds.

This backend intentionally avoids Open3D. It writes a standalone HTML file with
embedded WebGL code and downsampled point buffers, so the common debug loop is:

    wato_lidar_preprocessing viz --bag BAG --chunk 0000 --backend html

Then open the emitted file in any browser. The viewer is focused on the most
important inspection path for this stage: static vs dynamic point clouds, with
sweep scrubbing for dynamic points.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import subprocess
import webbrowser

import numpy as np

log = logging.getLogger(__name__)

_MAX_STATIC_HTML_PTS = 350_000
_MAX_DYNAMIC_HTML_PTS = 650_000
_STATIC_RGB = np.array([74, 143, 217], dtype=np.uint8)
_DYNAMIC_RGB = np.array([232, 77, 61], dtype=np.uint8)


def _sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, max_points, replace=False).astype(np.int64))


def _array_spec(arr: np.ndarray, dtype: np.dtype | str) -> dict:
    packed = np.ascontiguousarray(arr.astype(dtype, copy=False))
    return {
        "dtype": packed.dtype.name,
        "shape": list(packed.shape),
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _empty_spec(shape: tuple[int, ...], dtype: np.dtype | str) -> dict:
    return _array_spec(np.zeros(shape, dtype=dtype), dtype)


def _default_output_path(bag_id: str, chunk_id: str, sweep_id: int | None) -> Path:
    from wato_common.artifact_store import chunk_root, local_path

    stem = f"sweep_{sweep_id:06d}" if sweep_id is not None else "chunk"
    return Path(local_path(chunk_root(bag_id, chunk_id))) / "viz" / f"{stem}.html"


def _is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def open_html_file(path: str | Path) -> bool:
    """Open an HTML file in the default browser, resolving relative paths."""
    resolved = Path(path).resolve()
    if webbrowser.open(resolved.as_uri(), new=2):
        return True

    if not _is_wsl():
        return False

    try:
        converted = subprocess.run(
            ["wslpath", "-w", str(resolved)],
            check=False,
            capture_output=True,
            text=True,
        )
        target = (
            converted.stdout.strip() if converted.returncode == 0 else str(resolved)
        )
        return subprocess.run(["explorer.exe", target], check=False).returncode == 0
    except OSError:
        return False


def _load_chunk_payload(bag_id: str, chunk_id: str) -> dict:
    from wato_lidar_preprocessing.viz_data import load_chunk_viz_data

    data = load_chunk_viz_data(bag_id, chunk_id)
    static_xyz = data.static_xyz
    dynamic_xyz = data.dynamic_xyz
    sweep_id = data.dynamic_sweep_id
    intensity = data.dynamic_intensity

    n_static_total = len(static_xyz)
    n_dynamic_total = len(dynamic_xyz)
    static_idx = _sample_indices(n_static_total, _MAX_STATIC_HTML_PTS, seed=17)
    dynamic_idx = _sample_indices(n_dynamic_total, _MAX_DYNAMIC_HTML_PTS, seed=23)

    static_xyz = static_xyz[static_idx].astype(np.float32, copy=False)
    dynamic_xyz = dynamic_xyz[dynamic_idx].astype(np.float32, copy=False)
    sweep_id = sweep_id[dynamic_idx].astype(np.int32, copy=False)
    if intensity is not None:
        intensity = intensity[dynamic_idx].astype(np.float32, copy=False)

    if len(sweep_id) > 0:
        order = np.argsort(sweep_id, kind="stable")
        dynamic_xyz = dynamic_xyz[order]
        sweep_id = sweep_id[order]
        if intensity is not None:
            intensity = intensity[order]

    xyz_for_bounds = [a for a in (static_xyz, dynamic_xyz) if len(a) > 0]
    bounds_src = np.vstack(xyz_for_bounds) if xyz_for_bounds else np.zeros((1, 3))

    return {
        "title": f"lidar_preprocessing chunk {chunk_id}",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "mode": "chunk",
        "static_xyz": _array_spec(static_xyz, np.float32),
        "dynamic_xyz": _array_spec(dynamic_xyz, np.float32),
        "dynamic_sweep_id": _array_spec(sweep_id, np.int32),
        "dynamic_intensity": (
            _array_spec(intensity, np.float32)
            if intensity is not None
            else _empty_spec((0,), np.float32)
        ),
        "has_intensity": intensity is not None,
        "bounds_min": bounds_src.min(axis=0).astype(float).tolist(),
        "bounds_max": bounds_src.max(axis=0).astype(float).tolist(),
        "counts": {
            "static": int(len(static_xyz)),
            "dynamic": int(len(dynamic_xyz)),
            "static_sampled_from": int(n_static_total),
            "dynamic_sampled_from": int(n_dynamic_total),
        },
    }


def _load_sweep_payload(bag_id: str, chunk_id: str, sweep_id: int) -> dict:
    from wato_lidar_preprocessing.viz_data import load_sweep_viz_data

    data = load_sweep_viz_data(bag_id, chunk_id, sweep_id)
    xyz = data.xyz.astype(np.float32)
    dynamic = data.dynamic

    idx = _sample_indices(len(xyz), _MAX_DYNAMIC_HTML_PTS, seed=31)
    xyz = xyz[idx]
    dynamic = dynamic[idx]
    sweep_ids = np.full((len(xyz),), sweep_id, dtype=np.int32)
    colors = np.repeat(_STATIC_RGB[None, :], len(xyz), axis=0)
    colors[dynamic] = _DYNAMIC_RGB

    return {
        "title": f"lidar_preprocessing chunk {chunk_id} sweep {sweep_id}",
        "bag_id": bag_id,
        "chunk_id": chunk_id,
        "sweep_id": sweep_id,
        "mode": "sweep",
        "static_xyz": _empty_spec((0, 3), np.float32),
        "dynamic_xyz": _array_spec(xyz, np.float32),
        "dynamic_sweep_id": _array_spec(sweep_ids, np.int32),
        "dynamic_intensity": _empty_spec((0,), np.float32),
        "dynamic_rgb": _array_spec(colors, np.uint8),
        "has_intensity": False,
        "bounds_min": xyz.min(axis=0).astype(float).tolist() if len(xyz) else [0, 0, 0],
        "bounds_max": xyz.max(axis=0).astype(float).tolist() if len(xyz) else [1, 1, 1],
        "counts": {
            "static": int((~dynamic).sum()),
            "dynamic": int(dynamic.sum()),
            "sampled_from": int(len(idx)),
        },
    }


def write_html_viewer(
    bag_id: str,
    chunk_id: str,
    *,
    sweep_id: int | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Write a standalone HTML/WebGL point-cloud viewer and return its path."""
    payload = (
        _load_sweep_payload(bag_id, chunk_id, sweep_id)
        if sweep_id is not None
        else _load_chunk_payload(bag_id, chunk_id)
    )
    out = (
        Path(out_path)
        if out_path is not None
        else _default_output_path(bag_id, chunk_id, sweep_id)
    )
    if out.suffix.lower() != ".html":
        out.mkdir(parents=True, exist_ok=True)
        name = f"sweep_{sweep_id:06d}.html" if sweep_id is not None else "chunk.html"
        out = out / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_html(payload), encoding="utf-8")
    return out


def _render_html(payload: dict) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{payload["title"]}</title>
<style>
html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: #101113; color: #eef1f4; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
#gl {{ position: fixed; inset: 0; width: 100%; height: 100%; display: block; }}
#panel {{ position: fixed; left: 16px; right: 16px; bottom: 16px; display: grid; grid-template-columns: 1fr; gap: 10px; padding: 12px; background: rgba(18, 20, 23, 0.88); border: 1px solid rgba(255,255,255,0.14); border-radius: 8px; backdrop-filter: blur(8px); }}
#row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
#title {{ font-size: 14px; font-weight: 650; min-width: 260px; }}
#meta {{ color: #b8c0cc; font-size: 12px; }}
label {{ color: #dbe2ea; font-size: 12px; display: inline-flex; align-items: center; gap: 6px; }}
select, button, input[type="range"] {{ accent-color: #6fb7ff; }}
button, select {{ height: 30px; color: #eef1f4; background: #22272d; border: 1px solid #3b4651; border-radius: 6px; }}
button {{ padding: 0 10px; cursor: pointer; }}
input[type="range"] {{ min-width: 180px; }}
</style>
</head>
<body>
<canvas id="gl"></canvas>
<div id="panel">
  <div id="row">
    <div id="title">{payload["title"]}</div>
    <div id="meta"></div>
  </div>
  <div id="row">
    <label>view <select id="view"><option value="top">top</option><option value="iso">iso</option><option value="side">side</option><option value="front">front</option></select></label>
    <label>mode <select id="mode"><option value="single">single sweep</option><option value="trail">trail</option><option value="all">all dynamic</option></select></label>
    <label>color <select id="color"><option value="uniform">static/dynamic</option><option value="sweep">sweep id</option><option value="height">height</option><option value="intensity">intensity</option></select></label>
    <label>sweep <input id="sweep" type="range" min="0" max="0" value="0"></label>
    <button id="prev">prev</button>
    <button id="play">play</button>
    <button id="next">next</button>
    <label><input id="static" type="checkbox" checked> static</label>
    <label><input id="dynamic" type="checkbox" checked> dynamic</label>
    <label>point <input id="point" type="range" min="1" max="8" value="2"></label>
  </div>
</div>
<script>
const PAYLOAD = {payload_json};
const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl', {{ antialias: true, preserveDrawingBuffer: true }});
if (!gl) throw new Error('WebGL is not available in this browser');

function decode(spec, Ctor) {{
  const bin = atob(spec.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Ctor(bytes.buffer);
}}
const staticXYZ = decode(PAYLOAD.static_xyz, Float32Array);
const dynamicXYZ = decode(PAYLOAD.dynamic_xyz, Float32Array);
const dynamicSweep = decode(PAYLOAD.dynamic_sweep_id, Int32Array);
const dynamicIntensity = decode(PAYLOAD.dynamic_intensity, Float32Array);
const fixedRGB = PAYLOAD.dynamic_rgb ? decode(PAYLOAD.dynamic_rgb, Uint8Array) : null;

const sweepValues = Array.from(new Set(Array.from(dynamicSweep))).sort((a, b) => a - b);
const sweepInput = document.getElementById('sweep');
sweepInput.max = Math.max(0, sweepValues.length - 1);
const modeInput = document.getElementById('mode');
if (PAYLOAD.mode === 'sweep') modeInput.value = 'all';
const colorInput = document.getElementById('color');
if (!PAYLOAD.has_intensity) {{
  for (const opt of Array.from(colorInput.options)) if (opt.value === 'intensity') opt.disabled = true;
}}

function shader(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}}
const program = gl.createProgram();
gl.attachShader(program, shader(gl.VERTEX_SHADER, `
attribute vec3 aPos;
attribute vec3 aColor;
uniform mat4 uMvp;
uniform float uPointSize;
varying vec3 vColor;
void main() {{
  gl_Position = uMvp * vec4(aPos, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor;
}}`));
gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
precision mediump float;
varying vec3 vColor;
void main() {{
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}}`));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
gl.useProgram(program);
const locPos = gl.getAttribLocation(program, 'aPos');
const locColor = gl.getAttribLocation(program, 'aColor');
const locMvp = gl.getUniformLocation(program, 'uMvp');
const locPoint = gl.getUniformLocation(program, 'uPointSize');

function makeCloud(xyz, colors) {{
  const pos = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pos);
  gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.STATIC_DRAW);
  const col = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, col);
  gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
  return {{ pos, col, count: xyz.length / 3 }};
}}

function fillColor(n, rgb) {{
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) out.set(rgb, i * 3);
  return out;
}}
const staticCloud = makeCloud(staticXYZ, fillColor(staticXYZ.length / 3, [0.29, 0.56, 0.85]));
let dynamicCloud = makeCloud(new Float32Array(0), new Float32Array(0));

function ramp(t) {{
  t = Math.max(0, Math.min(1, t));
  return [Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 3))),
          Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 2))),
          Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 1)))];
}}
function dynamicSelection() {{
  const mode = modeInput.value;
  const current = sweepValues[Number(sweepInput.value)] ?? 0;
  const idx = [];
  for (let i = 0; i < dynamicSweep.length; i++) {{
    const sid = dynamicSweep[i];
    if (mode === 'all' || sid === current || (mode === 'trail' && sid <= current && sid > current - 5)) idx.push(i);
  }}
  const xyz = new Float32Array(idx.length * 3);
  const col = new Float32Array(idx.length * 3);
  let minI = Infinity, maxI = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const i of idx) {{
    if (dynamicIntensity.length) {{ minI = Math.min(minI, dynamicIntensity[i]); maxI = Math.max(maxI, dynamicIntensity[i]); }}
    const z = dynamicXYZ[i*3 + 2]; minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
  }}
  const cMode = colorInput.value;
  const sMin = sweepValues[0] ?? 0, sMax = sweepValues[sweepValues.length - 1] ?? 1;
  for (let j = 0; j < idx.length; j++) {{
    const i = idx[j];
    xyz[j*3] = dynamicXYZ[i*3]; xyz[j*3+1] = dynamicXYZ[i*3+1]; xyz[j*3+2] = dynamicXYZ[i*3+2];
    let c = [0.91, 0.30, 0.24];
    if (fixedRGB && cMode === 'uniform') c = [fixedRGB[i*3]/255, fixedRGB[i*3+1]/255, fixedRGB[i*3+2]/255];
    else if (cMode === 'sweep') c = ramp((dynamicSweep[i] - sMin) / Math.max(1, sMax - sMin));
    else if (cMode === 'height') c = ramp((dynamicXYZ[i*3+2] - minZ) / Math.max(0.001, maxZ - minZ));
    else if (cMode === 'intensity' && dynamicIntensity.length) c = ramp((dynamicIntensity[i] - minI) / Math.max(0.001, maxI - minI));
    col.set(c, j*3);
  }}
  dynamicCloud = makeCloud(xyz, col);
  document.getElementById('meta').textContent = `${{PAYLOAD.bag_id}} / ${{PAYLOAD.chunk_id}} · dynamic ${{idx.length.toLocaleString()}} pts · static ${{(staticXYZ.length/3).toLocaleString()}} pts`;
}}

function mat4Ortho(l, r, b, t, n, f) {{
  return [2/(r-l),0,0,0, 0,2/(t-b),0,0, 0,0,-2/(f-n),0, -(r+l)/(r-l),-(t+b)/(t-b),-(f+n)/(f-n),1];
}}
function normalize(v) {{ const d = Math.hypot(v[0],v[1],v[2]) || 1; return [v[0]/d,v[1]/d,v[2]/d]; }}
function cross(a,b) {{ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }}
function dot(a,b) {{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }}
function lookAt(eye, center, up) {{
  const z = normalize([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,eye),-dot(y,eye),-dot(z,eye),1];
}}
function mul(a,b) {{
  const o = new Array(16).fill(0);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) for (let k=0;k<4;k++) o[c*4+r]+=a[k*4+r]*b[c*4+k];
  return o;
}}
const mn = PAYLOAD.bounds_min, mx = PAYLOAD.bounds_max;
const center = [(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2];
const extent = Math.max(1, Math.hypot(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]));
function mvp() {{
  const aspect = canvas.width / Math.max(1, canvas.height);
  const half = extent * 0.58;
  const view = document.getElementById('view').value;
  const dist = extent * 1.8;
  let eye = [center[0], center[1], center[2] + dist], up = [0, 1, 0];
  if (view === 'iso') {{ eye = [center[0]+dist, center[1]-dist, center[2]+dist]; up = [0,0,1]; }}
  if (view === 'side') {{ eye = [center[0], center[1]-dist, center[2]]; up = [0,0,1]; }}
  if (view === 'front') {{ eye = [center[0]-dist, center[1], center[2]]; up = [0,0,1]; }}
  return mul(mat4Ortho(-half*aspect, half*aspect, -half, half, -dist*4, dist*4), lookAt(eye, center, up));
}}
function drawCloud(cloud) {{
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.pos);
  gl.vertexAttribPointer(locPos, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(locPos);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.col);
  gl.vertexAttribPointer(locColor, 3, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(locColor);
  gl.drawArrays(gl.POINTS, 0, cloud.count);
}}
function render() {{
  const dpr = window.devicePixelRatio || 1;
  const w = Math.floor(canvas.clientWidth*dpr), h = Math.floor(canvas.clientHeight*dpr);
  if (canvas.width !== w || canvas.height !== h) {{ canvas.width = w; canvas.height = h; gl.viewport(0,0,w,h); }}
  gl.clearColor(0.06, 0.065, 0.075, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(locMvp, false, new Float32Array(mvp()));
  gl.uniform1f(locPoint, Number(document.getElementById('point').value) * (window.devicePixelRatio || 1));
  if (document.getElementById('static').checked) drawCloud(staticCloud);
  if (document.getElementById('dynamic').checked) drawCloud(dynamicCloud);
  requestAnimationFrame(render);
}}
function refresh() {{ dynamicSelection(); }}
for (const id of ['view','mode','color','sweep','static','dynamic','point']) document.getElementById(id).addEventListener('input', refresh);
document.getElementById('prev').onclick = () => {{ sweepInput.value = Math.max(0, Number(sweepInput.value)-1); refresh(); }};
document.getElementById('next').onclick = () => {{ sweepInput.value = Math.min(Number(sweepInput.max), Number(sweepInput.value)+1); refresh(); }};
let timer = null;
document.getElementById('play').onclick = (ev) => {{
  if (timer) {{ clearInterval(timer); timer = null; ev.target.textContent = 'play'; return; }}
  ev.target.textContent = 'pause';
  timer = setInterval(() => {{ sweepInput.value = (Number(sweepInput.value)+1) % (Number(sweepInput.max)+1); refresh(); }}, 140);
}};
refresh();
render();
</script>
</body>
</html>
"""
