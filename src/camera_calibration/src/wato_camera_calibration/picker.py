"""Browser bundle for hand-picking independent static correspondences.

A correspondence pairs a pixel clicked on a held-out (validation/test) image
with a world point clicked in the static LiDAR cloud around that image's time.
The two sides are picked separately and LiDAR is never drawn on the image: a
pick guided by a projection would inherit the calibration it is meant to judge.
The recorded mount only sets the 3D view's starting viewpoint.

The bundle is static files under `camera_calibration/picker/`; serve the
`camera_calibration/` directory over HTTP and the page saves picks in the
format `validation.correspondences` reads.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from wato_common.artifacts import calibration_refinement as artifact
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.lidar_sweeps import read_dynamic_mask, read_world

from wato_camera_calibration.inputs import relocated, voxel_mean, world_to_ego
from wato_camera_calibration.sampling import observations, sweeps_near

log = logging.getLogger(__name__)

HELD_OUT = ("validation", "test")


def world_cloud(
    frame,
    rows: list,
    world_T_cam: np.ndarray,
    model,
    size: tuple[int, int],
    *,
    max_range_m: float,
    voxel_m: float,
    max_points: int,
):
    """Static returns in front of the camera, in world coordinates.

    Sweeps are already deskewed into the world frame, so accumulating them over
    a window densifies static structure without a per-sweep pose correction.
    """
    xyzs, values = [], []
    for row in rows:
        world = read_world(str(relocated(row.world_path)))
        mask = read_dynamic_mask(
            str(relocated(row.dynamic_mask_path)), n_points=world.n_points
        )
        if world.intensity is None:
            raise ValueError(f"{row.world_path}: intensity is missing")
        keep = ~mask & np.isfinite(world.xyz).all(axis=1) & np.isfinite(world.intensity)
        xyzs.append(world.xyz[keep])
        values.append(world.intensity[keep])
    if not xyzs:
        raise ValueError("no reference LiDAR sweeps near the image")
    xyz, intensity = np.concatenate(xyzs), np.concatenate(values)
    ego = world_to_ego(xyz, frame.world_T_ego)
    near = np.linalg.norm(ego, axis=1) <= max_range_m
    xyz, intensity = xyz[near], intensity[near]
    # A generous margin around the image keeps context for navigating the 3D
    # view; points far outside it cannot be matched to a pixel anyway.
    uv, depth, _ = model.project(xyz, np.linalg.inv(world_T_cam))
    width, height = size
    margin = 0.5 * max(width, height)
    front = (
        (depth > 0.3)
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= -margin)
        & (uv[:, 0] < width + margin)
        & (uv[:, 1] >= -margin)
        & (uv[:, 1] < height + margin)
    )
    xyz, intensity = xyz[front], intensity[front]
    if not len(xyz):
        raise ValueError("no static returns in front of the camera")
    xyz, intensity = voxel_mean(xyz, intensity, voxel_m)
    if len(xyz) > max_points:
        take = np.sort(
            np.random.default_rng(frame.timestamp_ns).choice(
                len(xyz), max_points, replace=False
            )
        )
        xyz, intensity = xyz[take], intensity[take]
    low, high = np.percentile(intensity, [1, 99])
    shade = np.clip((intensity - low) / max(high - low, 1e-9), 0, 1)
    return xyz, np.rint(shade * 255).astype(np.uint8)


def export(
    bag_id: str,
    *,
    cameras=None,
    window_s: float = 1.0,
    max_range_m: float = 60.0,
    voxel_m: float = 0.03,
    max_points: int = 400_000,
) -> Path:
    """Write the picker for every held-out view in the current report.

    Picks are keyed on the report's sampled image timestamps, so the run that
    later reads them must use the same sampling configuration and inputs.
    """
    _, report, manifest = artifact.read(bag_id)
    settings = manifest["metadata"]["settings"]["config"]
    lidar_id = settings["sampling"]["lidar_id"]
    recorded = Calibration.load(bag_id)
    root = artifact.directory(bag_id) / "picker"
    requested = sorted(
        cid
        for cid, row in report["cameras"].items()
        if row.get("sample_index") and (not cameras or cid in cameras)
    )
    if cameras and set(cameras) - set(requested):
        raise ValueError(
            f"no sampled views for {sorted(set(cameras) - set(requested))}; "
            "run camera_calibration (--prepare is enough) for them first"
        )
    if not requested:
        raise ValueError("the report has no sampled views; run camera_calibration")
    for cid in requested:
        if (root / cid).exists():
            shutil.rmtree(root / cid)
    index = {
        "bag_id": bag_id,
        "lidar_id": lidar_id,
        "window_s": window_s,
        "requirements": {
            "min_correspondences": settings["validation"]["min_correspondences"],
            "min_blocks_per_split": settings["validation"]["min_blocks_per_split"],
        },
        "cameras": {},
    }
    for cid in requested:
        cam = recorded.camera(cid)
        frames, rows, _ = observations(bag_id, cid, lidar_id)
        by_time = {f.timestamp_ns: f for f in frames}
        out = root / cid
        out.mkdir(parents=True, exist_ok=True)
        views = []
        for sample in report["cameras"][cid]["sample_index"]:
            if sample["split"] not in HELD_OUT:
                continue
            frame = by_time.get(int(sample["timestamp_ns"]))
            if frame is None:
                raise ValueError(
                    f"{cid}: sampled image {sample['timestamp_ns']} is no longer "
                    "in frame_index; rerun camera_calibration --force"
                )
            source = relocated(frame.image_path)
            with Image.open(source) as image:
                size = image.size
                model = cam.model_for_image(size)
            world_T_cam = frame.world_T_ego @ cam.ego_T_cam
            xyz, shade = world_cloud(
                frame,
                sweeps_near(rows, frame.timestamp_ns, window_s),
                world_T_cam,
                model,
                size,
                max_range_m=max_range_m,
                voxel_m=voxel_m,
                max_points=max_points,
            )
            # float32 world coordinates lose millimetres hundreds of metres
            # from the origin; store offsets from the camera centre instead.
            origin = world_T_cam[:3, 3].copy()
            stem = str(frame.timestamp_ns)
            shutil.copyfile(source, out / f"{stem}{source.suffix}")
            with open(out / f"{stem}.bin", "wb") as stream:
                stream.write((xyz - origin).astype("<f4").tobytes())
                stream.write(shade.tobytes())
            views.append(
                {
                    # Strings: nanosecond timestamps exceed JavaScript's
                    # exact-integer range.
                    "timestamp_ns": stem,
                    "split": sample["split"],
                    "block": sample["block"],
                    "image": f"{cid}/{stem}{source.suffix}",
                    "cloud": f"{cid}/{stem}.bin",
                    "n_points": len(xyz),
                    "origin": origin.tolist(),
                    "world_R_cam": world_T_cam[:3, :3].tolist(),
                }
            )
            log.info("%s %s: %d static points", cid, stem, len(xyz))
        if not views:
            log.warning("%s: no validation/test views to pick on", cid)
            continue
        index["cameras"][cid] = {
            "width": size[0],
            "height": size[1],
            "fy": float(cam.K[1, 1]),
            "views": views,
        }
    # Keep cameras exported earlier only while the report still samples
    # exactly their views; otherwise their picks would name unscored images.
    existing = root / "views.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get("bag_id") == bag_id:
            for cid, value in previous["cameras"].items():
                current = {
                    str(s["timestamp_ns"])
                    for s in report["cameras"].get(cid, {}).get("sample_index", [])
                    if s["split"] in HELD_OUT
                }
                kept = {v["timestamp_ns"] for v in value["views"]}
                if (
                    cid not in index["cameras"]
                    and (root / cid).exists()
                    and kept == current
                ):
                    index["cameras"][cid] = value
    index["cameras"] = dict(sorted(index["cameras"].items()))
    artifact.atomic_json(root / "views.json", index)
    (root / "index.html").write_text(PAGE, encoding="utf-8")
    return root


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration check points</title>
<style>
:root { --bg:#101113; --panel:#181b1f; --line:#2f363e; --text:#eef1f4; --muted:#9aa4b0; --accent:#6fb7ff; --pending:#33e0ff; --saved:#ffd23f; --group:#ff5fd2; --ok:#5fd38d; --warn:#ffb454; }
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--text); font:13px system-ui, -apple-system, Segoe UI, sans-serif; }
body { display:grid; grid-template-rows:auto 1fr auto; }
header, footer { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:8px 12px; background:var(--panel); border-bottom:1px solid var(--line); }
footer { border-top:1px solid var(--line); border-bottom:0; align-items:flex-start; max-height:32vh; overflow:auto; }
main { display:grid; grid-template-columns:1fr 1fr; min-height:0; }
.pane { position:relative; min-width:0; min-height:0; border-right:1px solid var(--line); overflow:hidden; }
.pane canvas { position:absolute; inset:0; width:100%; height:100%; display:block; }
.tag { position:absolute; left:8px; top:8px; padding:3px 7px; background:rgba(0,0,0,.65); border-radius:4px; color:var(--muted); pointer-events:none; }
#labels { position:absolute; inset:0; pointer-events:none; }
#labels span { position:absolute; transform:translate(6px,-18px); color:#111; background:var(--saved); border-radius:3px; padding:0 4px; font-weight:650; font-size:11px; }
select, button, input { height:28px; color:var(--text); background:#22272d; border:1px solid #3b4651; border-radius:6px; padding:0 8px; font:inherit; }
button { cursor:pointer; } button:hover { border-color:var(--accent); }
button:disabled { opacity:.4; cursor:default; border-color:#3b4651; }
button.primary { background:#1d4f7a; border-color:#2f73ad; }
label { color:var(--muted); display:inline-flex; gap:6px; align-items:center; }
#need { color:var(--muted); } .ok { color:var(--ok); } .warn { color:var(--warn); }
#pending { min-width:320px; } #pending b { color:var(--pending); font-weight:600; }
#pairs { display:flex; flex-wrap:wrap; gap:6px; }
#pairs div { background:#22272d; border:1px solid var(--line); border-radius:6px; padding:3px 6px; display:flex; gap:6px; align-items:center; }
#pairs button { height:20px; padding:0 6px; }
#help { color:var(--muted); flex-basis:100%; }
@media (max-width: 900px) { main { grid-template-columns:1fr; grid-template-rows:1fr 1fr; } }
</style>
</head>
<body>
<header>
  <label>camera <select id="cam"></select></label>
  <label>view <select id="view"></select></label>
  <button id="prev" title="previous view ([)">&larr;</button><button id="next" title="next view (])">&rarr;</button>
  <span id="need"></span>
  <span style="flex:1"></span>
  <label>colour <select id="colour"><option value="intensity">intensity</option><option value="height">height</option><option value="range">range</option></select></label>
  <label>point <input id="size" type="range" min="1" max="8" value="3" style="width:90px"></label>
  <button id="reset" title="back to the camera's viewpoint (R)">reset 3D view</button>
  <button id="load">load JSON&hellip;</button>
  <button id="save" class="primary">download correspondences.json</button>
  <input id="file" type="file" accept=".json,application/json" hidden>
</header>
<main>
  <div class="pane" id="imgPane"><canvas id="img"></canvas><div class="tag">image: click to place a pixel &middot; wheel zoom &middot; drag pan</div></div>
  <div class="pane" id="pcPane"><canvas id="gl"></canvas><div id="labels"></div><div class="tag">LiDAR (static, world): click a point &middot; shift-click adds to an average &middot; drag orbit &middot; right/ctrl-drag pan &middot; wheel zoom &middot; double-click centres</div></div>
</main>
<footer>
  <div id="pending"></div>
  <button id="add" class="primary" title="Enter">add pair</button>
  <button id="clear" title="Esc">clear pending</button>
  <div id="pairs"></div>
  <div id="help">Pick sharp static features you can place exactly in both: pole bases, sign or building corners, lane-marking corners. Spread them over the image and over near and far depths. LiDAR is never drawn on the image, so a pick cannot be steered by the calibration it will judge.</div>
</footer>
<script>
"use strict";
const $ = (id) => document.getElementById(id);
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
let meta, picks = {}, cam = null, view = null, cloud = null, image = null;
let pending = { pixel: null, group: [] };

// ---------- persistence: picks[cam][timestamp] = [{pixel:[u,v], world:[x,y,z], n:k}]
function draftKey() { return "wato-calib-picks:" + meta.bag_id; }
function persist() { store.set(draftKey(), JSON.stringify(picks)); updateNeed(); }
function fromFile(data) {
  if (data.schema_version !== 1 || data.bag_id !== meta.bag_id) throw new Error("not a schema 1 file for " + meta.bag_id);
  const out = {};
  for (const [cid, rows] of Object.entries(data.cameras || {})) {
    out[cid] = {};
    for (const row of rows) {
      const ts = String(row.timestamp_ns);
      out[cid][ts] = (out[cid][ts] || []).concat(row.world_points.map((w, i) => ({ world: w, pixel: row.pixels[i], n: 1 })));
    }
  }
  return out;
}
function toFile() {
  const cameras = {};
  for (const [cid, byTime] of Object.entries(picks)) {
    const rows = Object.entries(byTime).filter(([, list]) => list.length).map(([ts, list]) => ({
      timestamp_ns: "@@" + ts + "@@", provenance: "independent_static",
      world_points: list.map((p) => p.world), pixels: list.map((p) => p.pixel) }));
    if (rows.length) cameras[cid] = rows;
  }
  // Timestamps go out as integers, which JSON.stringify cannot write exactly.
  return JSON.stringify({ schema_version: 1, bag_id: meta.bag_id, cameras }, null, 1)
    .replace(/"@@(\d+)@@"/g, "$1");
}

// ---------- requirement counts, mirroring validation.evaluate's evidence gate
function updateNeed() {
  if (!cam) return;
  const req = meta.requirements, parts = [];
  for (const split of ["validation", "test"]) {
    let n = 0; const blocks = new Set();
    for (const v of meta.cameras[cam].views) {
      if (v.split !== split) continue;
      const k = ((picks[cam] || {})[v.timestamp_ns] || []).length;
      n += k; if (k) blocks.add(v.block);
    }
    const ok = n >= req.min_correspondences && blocks.size >= req.min_blocks_per_split;
    parts.push(`<span class="${ok ? "ok" : "warn"}">${split} ${n}/${req.min_correspondences} pts, ${blocks.size}/${req.min_blocks_per_split} blocks</span>`);
  }
  $("need").innerHTML = parts.join(" &middot; ");
  for (const opt of $("view").options) {
    const v = meta.cameras[cam].views[Number(opt.value)];
    const k = ((picks[cam] || {})[v.timestamp_ns] || []).length;
    opt.textContent = `${v.split} · block ${v.block} · ${v.timestamp_ns}${k ? ` · ${k} picked` : ""}`;
  }
}

// ---------- image pane
const ic = $("img"), ix = ic.getContext("2d");
let it = { s: 1, x: 0, y: 0 }, drag = null;
function fitImage() {
  const w = ic.clientWidth, h = ic.clientHeight;
  it.s = Math.min(w / image.width, h / image.height); it.x = (w - image.width * it.s) / 2; it.y = (h - image.height * it.s) / 2;
}
function drawImage() {
  const dpr = devicePixelRatio || 1, w = ic.clientWidth, h = ic.clientHeight;
  if (ic.width !== Math.round(w * dpr) || ic.height !== Math.round(h * dpr)) { ic.width = Math.round(w * dpr); ic.height = Math.round(h * dpr); }
  ix.setTransform(dpr, 0, 0, dpr, 0, 0); ix.fillStyle = "#0b0c0e"; ix.fillRect(0, 0, w, h);
  if (!image) return;
  ix.imageSmoothingEnabled = it.s < 2;
  ix.drawImage(image, it.x, it.y, image.width * it.s, image.height * it.s);
  // Pixel (u, v) is a pixel centre in K's convention; its area starts at u - 0.5.
  const at = (p) => [(p[0] + 0.5) * it.s + it.x, (p[1] + 0.5) * it.s + it.y];
  ((picks[cam] || {})[view.timestamp_ns] || []).forEach((p, i) => {
    const [x, y] = at(p.pixel); ix.strokeStyle = "#ffd23f"; ix.lineWidth = 1.5;
    ix.beginPath(); ix.arc(x, y, 6, 0, 7); ix.stroke();
    ix.fillStyle = "#ffd23f"; ix.font = "600 12px system-ui"; ix.fillText(String(i + 1), x + 8, y - 8);
  });
  if (pending.pixel) {
    const [x, y] = at(pending.pixel); ix.strokeStyle = "#33e0ff"; ix.lineWidth = 1;
    ix.beginPath(); ix.moveTo(x - 14, y); ix.lineTo(x - 3, y); ix.moveTo(x + 3, y); ix.lineTo(x + 14, y);
    ix.moveTo(x, y - 14); ix.lineTo(x, y - 3); ix.moveTo(x, y + 3); ix.lineTo(x, y + 14); ix.stroke();
  }
}
ic.addEventListener("wheel", (e) => {
  e.preventDefault(); const r = ic.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const f = Math.pow(1.2, -e.deltaY / 100), s = Math.min(64, Math.max(0.05, it.s * f));
  it.x = mx - (mx - it.x) * (s / it.s); it.y = my - (my - it.y) * (s / it.s); it.s = s; drawImage();
}, { passive: false });
ic.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, ox: it.x, oy: it.y, moved: false }; });
addEventListener("mousemove", (e) => {
  if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  if (Math.hypot(dx, dy) > 4) drag.moved = true;
  if (drag.moved) { it.x = drag.ox + dx; it.y = drag.oy + dy; drawImage(); }
});
addEventListener("mouseup", (e) => {
  if (!drag) return; const d = drag; drag = null;
  if (d.moved || e.target !== ic || !image) return;
  const r = ic.getBoundingClientRect();
  const u = (e.clientX - r.left - it.x) / it.s - 0.5, v = (e.clientY - r.top - it.y) / it.s - 0.5;
  if (u < -0.5 || v < -0.5 || u > image.width - 0.5 || v > image.height - 0.5) return;
  pending.pixel = [Math.round(u * 100) / 100, Math.round(v * 100) / 100]; showPending(); drawImage();
});

// ---------- 3D pane (Z up, offsets from the camera centre)
const gc = $("gl"), gl = gc.getContext("webgl", { antialias: true });
function sh(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }
const prog = gl.createProgram();
gl.attachShader(prog, sh(gl.VERTEX_SHADER, "attribute vec3 p; attribute vec3 c; uniform mat4 m; uniform float s; varying vec3 vc; void main(){ gl_Position=m*vec4(p,1.0); gl_PointSize=s; vc=c; }"));
gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, "precision mediump float; varying vec3 vc; void main(){ vec2 d=gl_PointCoord-vec2(.5); if(dot(d,d)>.25) discard; gl_FragColor=vec4(vc,1.0); }"));
gl.linkProgram(prog); gl.useProgram(prog);
const aP = gl.getAttribLocation(prog, "p"), aC = gl.getAttribLocation(prog, "c"), uM = gl.getUniformLocation(prog, "m"), uS = gl.getUniformLocation(prog, "s");
gl.enable(gl.DEPTH_TEST);
let buffers = null, overlay = null, orbit = null, mvp = null;
function ramp(t) { t = Math.max(0, Math.min(1, t)); return [Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3))), Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2))), Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)))]; }
function colours() {
  const n = cloud.n, out = new Float32Array(n * 3), mode = $("colour").value, xyz = cloud.xyz;
  for (let i = 0; i < n; i++) {
    let c;
    if (mode === "intensity") { const g = 0.12 + 0.88 * cloud.shade[i] / 255; c = [g, g, g]; }
    else if (mode === "height") c = ramp((xyz[i * 3 + 2] + 3) / 6);
    else c = ramp(Math.hypot(xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]) / 60);
    out.set(c, i * 3);
  }
  return out;
}
function upload(xyz, rgb) {
  const p = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, p); gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.STATIC_DRAW);
  const c = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, c); gl.bufferData(gl.ARRAY_BUFFER, rgb, gl.STATIC_DRAW);
  return { p, c, n: xyz.length / 3 };
}
function draw(b, size) {
  if (!b || !b.n) return; gl.uniform1f(uS, size);
  gl.bindBuffer(gl.ARRAY_BUFFER, b.p); gl.vertexAttribPointer(aP, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(aP);
  gl.bindBuffer(gl.ARRAY_BUFFER, b.c); gl.vertexAttribPointer(aC, 3, gl.FLOAT, false, 0, 0); gl.enableVertexAttribArray(aC);
  gl.drawArrays(gl.POINTS, 0, b.n);
}
function resetOrbit() {
  // Start at the camera centre looking along its optical axis (recorded mount;
  // orientation aid only, never part of a pick).
  const R = view.world_R_cam, f = [R[0][2], R[1][2], R[2][2]];
  orbit = { target: [f[0] * 10, f[1] * 10, f[2] * 10], dist: 10, yaw: Math.atan2(f[1], f[0]), pitch: Math.asin(Math.max(-1, Math.min(1, f[2]))) };
}
const norm = (v) => { const d = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / d, v[1] / d, v[2] / d]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function basis() {
  const f = [Math.cos(orbit.pitch) * Math.cos(orbit.yaw), Math.cos(orbit.pitch) * Math.sin(orbit.yaw), Math.sin(orbit.pitch)];
  const r = norm(cross(f, [0, 0, 1])), u = cross(r, f);
  const eye = [orbit.target[0] - f[0] * orbit.dist, orbit.target[1] - f[1] * orbit.dist, orbit.target[2] - f[2] * orbit.dist];
  return { f, r, u, eye };
}
function matrix(w, h) {
  const { f, r, u, eye } = basis();
  const view = [r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, eye), -dot(u, eye), dot(f, eye), 1];
  const c = meta.cameras[cam], fov = 2 * Math.atan(c.height / 2 / c.fy) * 1.15, t = 1 / Math.tan(fov / 2), n = 0.05, fa = 400;
  const proj = [t / (w / h), 0, 0, 0, 0, t, 0, 0, 0, 0, (fa + n) / (n - fa), -1, 0, 0, 2 * fa * n / (n - fa), 0];
  const o = new Array(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) o[i * 4 + j] += proj[k * 4 + j] * view[i * 4 + k];
  return o;
}
function toScreen(m, x, y, z, w, h) {
  const cx = m[0] * x + m[4] * y + m[8] * z + m[12], cy = m[1] * x + m[5] * y + m[9] * z + m[13];
  const cz = m[2] * x + m[6] * y + m[10] * z + m[14], cw = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (cw <= 0) return null;
  return [(cx / cw + 1) / 2 * w, (1 - cy / cw) / 2 * h, cz / cw];
}
function rebuildOverlay() {
  if (!cloud) return;
  const pts = [], rgb = [];
  ((picks[cam] || {})[view.timestamp_ns] || []).forEach((p) => { pts.push(...rel(p.world)); rgb.push(1, 0.82, 0.25); });
  for (const i of pending.group) { pts.push(cloud.xyz[i * 3], cloud.xyz[i * 3 + 1], cloud.xyz[i * 3 + 2]); rgb.push(1, 0.37, 0.82); }
  const pw = pendingWorld(); if (pw) { pts.push(...rel(pw)); rgb.push(0.2, 0.88, 1); }
  overlay = upload(new Float32Array(pts), new Float32Array(rgb));
}
const rel = (w) => [w[0] - view.origin[0], w[1] - view.origin[1], w[2] - view.origin[2]];
function render() {
  const dpr = devicePixelRatio || 1, w = gc.clientWidth, h = gc.clientHeight;
  if (gc.width !== Math.round(w * dpr) || gc.height !== Math.round(h * dpr)) { gc.width = Math.round(w * dpr); gc.height = Math.round(h * dpr); }
  gl.viewport(0, 0, gc.width, gc.height); gl.clearColor(0.04, 0.045, 0.05, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (cloud && orbit) {
    mvp = matrix(w, h); gl.uniformMatrix4fv(uM, false, new Float32Array(mvp));
    const s = Number($("size").value) * dpr; draw(buffers, s);
    gl.disable(gl.DEPTH_TEST); draw(overlay, s + 6 * dpr); gl.enable(gl.DEPTH_TEST);
    const labels = $("labels"); labels.innerHTML = "";
    ((picks[cam] || {})[view.timestamp_ns] || []).forEach((p, i) => {
      const q = toScreen(mvp, ...rel(p.world), w, h); if (!q) return;
      const el = document.createElement("span"); el.textContent = String(i + 1); el.style.left = q[0] + "px"; el.style.top = q[1] + "px"; labels.appendChild(el);
    });
  }
  drawImage(); requestAnimationFrame(render);
}
function pickPoint(e) {
  const r = gc.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, w = gc.clientWidth, h = gc.clientHeight;
  const radius = Math.max(4, Number($("size").value) / 2 + 3);
  let best = -1, bestZ = Infinity; const xyz = cloud.xyz;
  for (let i = 0; i < cloud.n; i++) {
    const q = toScreen(mvp, xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2], w, h);
    if (!q || Math.abs(q[0] - mx) > radius || Math.abs(q[1] - my) > radius || Math.hypot(q[0] - mx, q[1] - my) > radius) continue;
    if (q[2] < bestZ) { bestZ = q[2]; best = i; }   // the front-most visible disc
  }
  return best;
}
function pendingWorld() {
  if (!pending.group.length) return null;
  const s = [0, 0, 0];
  for (const i of pending.group) for (let k = 0; k < 3; k++) s[k] += cloud.xyz[i * 3 + k];
  return s.map((v, k) => v / pending.group.length + view.origin[k]);
}
let gdrag = null;
gc.addEventListener("contextmenu", (e) => e.preventDefault());
gc.addEventListener("mousedown", (e) => { gdrag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.ctrlKey || e.metaKey, moved: false, shift: e.shiftKey }; });
addEventListener("mousemove", (e) => {
  if (!gdrag) return; const dx = e.clientX - gdrag.x, dy = e.clientY - gdrag.y;
  if (!gdrag.moved && Math.hypot(dx, dy) < 4) return; gdrag.moved = true; gdrag.x = e.clientX; gdrag.y = e.clientY;
  if (gdrag.pan) {
    const { r, u } = basis(), k = orbit.dist * 0.0015;
    for (let i = 0; i < 3; i++) orbit.target[i] += (-dx * r[i] + dy * u[i]) * k;
  } else { orbit.yaw -= dx * 0.005; orbit.pitch = Math.max(-1.5, Math.min(1.5, orbit.pitch - dy * 0.005)); }
});
addEventListener("mouseup", (e) => {
  if (!gdrag) return; const d = gdrag; gdrag = null;
  if (d.moved || e.target !== gc || !cloud || e.button !== 0) return;
  const i = pickPoint(e); if (i < 0) return;
  if (d.shift) { const k = pending.group.indexOf(i); if (k >= 0) pending.group.splice(k, 1); else pending.group.push(i); }
  else pending.group = [i];
  rebuildOverlay(); showPending();
});
gc.addEventListener("dblclick", (e) => {
  const i = pickPoint(e); if (i < 0) return;
  const { eye } = basis(); orbit.target = [cloud.xyz[i * 3], cloud.xyz[i * 3 + 1], cloud.xyz[i * 3 + 2]];
  orbit.dist = Math.max(0.5, Math.hypot(eye[0] - orbit.target[0], eye[1] - orbit.target[1], eye[2] - orbit.target[2]));
});
gc.addEventListener("wheel", (e) => { e.preventDefault(); orbit.dist = Math.max(0.2, Math.min(300, orbit.dist * Math.pow(1.15, e.deltaY / 100))); }, { passive: false });

// ---------- pairing
function showPending() {
  const pw = pendingWorld(), px = pending.pixel;
  $("pending").innerHTML = `pending: pixel <b>${px ? px.map((v) => v.toFixed(2)).join(", ") : "—"}</b> &harr; LiDAR <b>${pw ? pw.map((v) => v.toFixed(3)).join(", ") + (pending.group.length > 1 ? ` (mean of ${pending.group.length})` : "") : "—"}</b>`;
  $("add").disabled = !(px && pw);
}
function clearPending() { pending = { pixel: null, group: [] }; rebuildOverlay(); showPending(); }
function addPair() {
  const pw = pendingWorld(); if (!pending.pixel || !pw) return;
  picks[cam] = picks[cam] || {}; const list = (picks[cam][view.timestamp_ns] = picks[cam][view.timestamp_ns] || []);
  list.push({ pixel: pending.pixel, world: pw.map((v) => Math.round(v * 1e4) / 1e4), n: pending.group.length });
  persist(); clearPending(); listPairs();
}
function listPairs() {
  const box = $("pairs"); box.innerHTML = "";
  ((picks[cam] || {})[view.timestamp_ns] || []).forEach((p, i) => {
    const el = document.createElement("div");
    el.innerHTML = `<b>${i + 1}</b> (${p.pixel.map((v) => v.toFixed(1)).join(", ")}) &harr; (${p.world.map((v) => v.toFixed(2)).join(", ")})`;
    const del = document.createElement("button"); del.textContent = "delete";
    del.onclick = () => { picks[cam][view.timestamp_ns].splice(i, 1); persist(); rebuildOverlay(); listPairs(); };
    el.appendChild(del); box.appendChild(el);
  });
}

// ---------- loading
async function loadView(index) {
  view = meta.cameras[cam].views[index]; $("view").value = String(index);
  pending = { pixel: null, group: [] }; cloud = null; image = null; showPending(); listPairs();
  const [img, buf] = await Promise.all([
    new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = bad; im.src = view.image; }),
    fetch(view.cloud).then((r) => { if (!r.ok) throw new Error(view.cloud + ": " + r.status); return r.arrayBuffer(); }),
  ]);
  if (meta.cameras[cam].views[index] !== view) return;
  image = img; fitImage();
  cloud = { n: view.n_points, xyz: new Float32Array(buf, 0, view.n_points * 3), shade: new Uint8Array(buf, view.n_points * 12, view.n_points) };
  buffers = upload(cloud.xyz, colours()); resetOrbit(); rebuildOverlay(); updateNeed();
}
function selectCamera(cid) {
  cam = cid; $("cam").value = cid; $("view").innerHTML = "";
  meta.cameras[cid].views.forEach((v, i) => { const o = document.createElement("option"); o.value = String(i); $("view").appendChild(o); });
  updateNeed(); loadView(0);
}
function step(d) { const n = meta.cameras[cam].views.length; loadView((Number($("view").value) + d + n) % n); }
async function start() {
  meta = await (await fetch("views.json", { cache: "no-store" })).json();
  const draft = store.get(draftKey());
  if (draft) picks = JSON.parse(draft);
  else {
    // An existing authored file beside the picker seeds a fresh browser.
    try { const r = await fetch("../correspondences.json", { cache: "no-store" }); if (r.ok) picks = fromFile(await r.json()); } catch (e) {}
  }
  for (const cid of Object.keys(meta.cameras)) { const o = document.createElement("option"); o.value = o.textContent = cid; $("cam").appendChild(o); }
  $("cam").onchange = (e) => selectCamera(e.target.value);
  $("view").onchange = (e) => loadView(Number(e.target.value));
  $("prev").onclick = () => step(-1); $("next").onclick = () => step(1);
  $("colour").onchange = () => { if (cloud) buffers = upload(cloud.xyz, colours()); };
  $("reset").onclick = () => view && resetOrbit();
  $("add").onclick = addPair; $("clear").onclick = clearPending;
  $("save").onclick = () => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([toFile()], { type: "application/json" }));
    a.download = "correspondences.json"; a.click(); URL.revokeObjectURL(a.href);
  };
  $("load").onclick = () => $("file").click();
  $("file").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try { const next = fromFile(JSON.parse(await f.text())); if (confirm("Replace the picks in this browser with the file's?")) { picks = next; persist(); rebuildOverlay(); listPairs(); } }
    catch (err) { alert(err.message); }
    e.target.value = "";
  };
  addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "Enter") addPair(); else if (e.key === "Escape") clearPending();
    else if (e.key === "[") step(-1); else if (e.key === "]") step(1); else if (e.key === "r" || e.key === "R") view && resetOrbit();
  });
  addEventListener("resize", () => image && fitImage());
  selectCamera(Object.keys(meta.cameras)[0]); requestAnimationFrame(render);
}
start().catch((e) => { document.body.innerHTML = `<pre style="padding:16px">${String(e.stack || e)}</pre>`; });
</script>
</body>
</html>
"""
