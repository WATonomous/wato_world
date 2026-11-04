"""Audit camera projection against fixed track returns and their own masks.

This is an offline diagnostic, not an automatic calibration stage. A mask is
not ground truth; containment alone cannot identify a physical calibration.
Fit each parameter family separately on 70% of track IDs, then evaluate on
the other IDs. Points, masks and denominators stay fixed throughout a sweep.
The CLI writes an experimental calibration beside its report; source artifacts
are never rewritten. See docs/camera_projection_measurements.md for limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from wato_common.artifact_store import artifact_root
from wato_common.artifacts import tracks
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.frame_index import FrameIndex
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps, read_world
from wato_common.artifacts.masklets import MaskletIndex
from wato_common.geometry import CameraModel, invert_se3
from wato_common.pose_lookup import PoseLookup
from wato_common.schemas import Box3D
from wato_common.tracking import points_in_box

from wato_semantic_lifting.projection import (
    compute_cam_T_world,
    pixel_indices,
    project_and_clip,
)


@dataclass
class Sample:
    """One fixed (track, image) correspondence, with a cropped mask distance field.

    offset_poses[k] maps ego coordinates at image time to ego coordinates at
    image time + OFFSETS[k]. NaN means the shifted pose is untrusted.
    """

    cam_id: str
    track_id: str
    chunk_id: str
    seq: int
    timestamp_ns: int
    sweep_id: int
    cls: str
    state: str
    train: bool
    xyz: np.ndarray
    ego_xyz: np.ndarray
    lidar_ids: np.ndarray
    offset_poses: np.ndarray
    speed: float
    mask: np.ndarray
    distance: np.ndarray
    origin: np.ndarray
    centroid: np.ndarray
    extent: np.ndarray
    image_path: str


OFFSETS = np.arange(-200.0, 201.0, 10.0)
FAMILIES = {
    "yaw": ([0.0], [(-3.0, 3.0)]),
    "pitch": ([0.0], [(-3.0, 3.0)]),
    "rotation": ([0.0, 0.0, 0.0], [(-3.0, 3.0)] * 3),
    "translation": ([0.0, 0.0, 0.0], [(-0.3, 0.3)] * 3),
    "principal": ([0.0, 0.0], [(-40.0, 40.0)] * 2),
    "focal": ([1.0], [(0.95, 1.05)]),
}


def relocated(path: str) -> str:
    # Index paths were recorded inside Docker; do not modify the source artifact.
    return (
        artifact_root() + path[len("file:///data/artifacts") :]
        if path.startswith("file:///data/artifacts/")
        else (
            artifact_root().removeprefix("file://") + path[len("/data/artifacts") :]
            if path.startswith("/data/artifacts/")
            else path
        )
    )


def training_track(track_id: str) -> bool:
    """Keep all views of a linked track in the same deterministic split."""
    return int(hashlib.sha256(track_id.encode()).hexdigest()[:8], 16) % 10 < 7


def collect(bag, max_per_cam, canonical_lidar, chunks=None):
    frames, summaries = tracks.read(bag)
    summary = {s.track_id: s for s in summaries}
    calib = Calibration.load(bag)
    counts, per_track = Counter(), Counter()
    seen = set()
    samples = []
    candidates = [
        f
        for f in frames
        if f.owned
        and summary[f.track_id].cls in ("car", "pedestrian")
        and (chunks is None or f.chunk_id in chunks)
    ]
    np.random.default_rng(0).shuffle(candidates)
    chunk_ids = sorted({f.chunk_id for f in candidates})
    indices = {c: FrameIndex.load(bag, c) for c in chunk_ids}
    masks = {c: MaskletIndex.load(bag, c) for c in chunk_ids}
    recs = {c: {r.masklet_id: r for r in masks[c].records} for c in chunk_ids}
    sweeps = {c: ProcessedSweeps.load(bag, c) for c in chunk_ids}
    grouped = {
        c: {f.reference_sweep_id: f for f in sweeps[c].frames(canonical_lidar)}
        for c in chunk_ids
    }
    poses = {c: PoseLookup.load(bag, c) for c in chunk_ids}

    @lru_cache(maxsize=24)
    def cloud(c, sid):
        fr = grouped[c][sid]
        worlds = [
            read_world(relocated(sweeps[c].get(s).world_path)) for s in fr.sweep_ids
        ]
        return np.concatenate([w.xyz for w in worlds]), np.concatenate(
            [
                np.full(w.n_points, sweeps[c].get(s).lidar_id)
                for s, w in zip(fr.sweep_ids, worlds)
            ]
        )

    for f in candidates:
        mine = [
            recs[f.chunk_id][m]
            for m in f.supporting_masklet_ids
            if m in recs[f.chunk_id]
        ]
        refs = []
        for cf in indices[f.chunk_id].for_sweep(f.sweep_id):
            key = (f.track_id, cf.cam_id, cf.timestamp_ns)
            if (
                key in seen
                or cf.world_T_ego is None
                or counts[cf.cam_id] >= max_per_cam
                or per_track[f.track_id, cf.cam_id] >= 12
            ):
                continue
            matching = [
                r
                for r in mine
                if r.cam_id == cf.cam_id and cf.camera_seq in r.frames_present
            ]
            if matching:
                refs.append((cf, matching))
        if not refs or f.sweep_id not in grouped[f.chunk_id]:
            continue
        xyz, lids = cloud(f.chunk_id, f.sweep_id)
        box = Box3D(cx=f.cx, cy=f.cy, cz=f.cz, w=f.w, l=f.l, h=f.h, heading=f.heading)
        keep = points_in_box(xyz, box, 0.0)
        xyz, lids = xyz[keep], lids[keep]
        if len(xyz) < 20:
            continue
        if len(xyz) > 400:
            take = np.sort(
                np.random.default_rng(0).choice(len(xyz), 400, replace=False)
            )
            xyz, lids = xyz[take], lids[take]
        for cf, matching in refs:
            cam = calib.camera(cf.cam_id)
            _, _, ok = project_and_clip(
                xyz,
                cam.model,
                compute_cam_T_world(cf.world_T_ego, cam.ego_T_cam),
                (cam.width, cam.height),
            )
            if ok.sum() < 20:
                continue
            mask = np.logical_or.reduce(
                [masks[f.chunk_id].mask(r, cf.camera_seq) for r in matching]
            )
            yy, xx = np.nonzero(mask)
            if len(xx) < 100:
                continue
            # Crop distance fields to the mask plus 100 px; out-of-crop points get
            # a fixed loss and remain in the denominator for containment.
            x0, y0 = max(0, int(xx.min()) - 100), max(0, int(yy.min()) - 100)
            x1, y1 = (
                min(cam.width, int(xx.max()) + 101),
                min(cam.height, int(yy.max()) + 101),
            )
            crop = mask[y0:y1, x0:x1]
            distances = distance_transform_edt(~crop).astype(np.float32)
            ego = xyz @ cf.world_T_ego[:3, :3] + invert_se3(cf.world_T_ego)[:3, 3]
            offset_poses = []
            for delta in OFFSETS:
                p = poses[f.chunk_id].at(cf.timestamp_ns + int(delta * 1e6))
                offset_poses.append(
                    invert_se3(p.world_T_ego) @ cf.world_T_ego
                    if p.valid
                    else np.full((4, 4), np.nan)
                )
            p0 = poses[f.chunk_id].at(cf.timestamp_ns - 50_000_000)
            p1 = poses[f.chunk_id].at(cf.timestamp_ns + 50_000_000)
            speed = (
                float(
                    np.linalg.norm(p1.world_T_ego[:3, 3] - p0.world_T_ego[:3, 3]) / 0.1
                )
                if p0.valid and p1.valid
                else 0.0
            )
            is_train = training_track(f.track_id)
            samples.append(
                Sample(
                    cf.cam_id,
                    f.track_id,
                    f.chunk_id,
                    cf.camera_seq,
                    cf.timestamp_ns,
                    f.sweep_id,
                    summary[f.track_id].cls,
                    summary[f.track_id].motion_state,
                    is_train,
                    xyz,
                    ego,
                    lids,
                    np.array(offset_poses),
                    speed,
                    crop,
                    distances,
                    np.array([x0, y0]),
                    np.array([xx.mean(), yy.mean()]),
                    np.array([max(np.ptp(xx), 1), max(np.ptp(yy), 1)]),
                    relocated(cf.image_path),
                )
            )
            counts[cf.cam_id] += 1
            per_track[f.track_id, cf.cam_id] += 1
            seen.add((f.track_id, cf.cam_id, cf.timestamp_ns))
            if len(samples) % 100 == 0:
                print("collected", len(samples), dict(counts), flush=True)
    print("collected", len(samples), "views", dict(counts), flush=True)
    return samples


def save_samples(path: Path, samples: list[Sample], metadata: dict) -> None:
    """An internal diagnostic cache: JSON metadata + numeric NPZs, no pickle."""
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, sample in enumerate(samples):
        values = {f.name: getattr(sample, f.name) for f in fields(Sample)}
        arrays = {key: v for key, v in values.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path / f"{k:06d}.npz", **arrays)
        rows.append({key: v for key, v in values.items() if key not in arrays})
    (path / "index.json").write_text(
        json.dumps({"version": 1, "metadata": metadata, "samples": rows}, indent=2)
        + "\n"
    )


def load_samples(path: Path, metadata: dict) -> list[Sample]:
    index = json.loads((path / "index.json").read_text())
    if index["version"] != 1 or index["metadata"] != metadata:
        raise ValueError("audit cache inputs changed; choose a new --cache directory")
    samples = []
    for k, row in enumerate(index["samples"]):
        with np.load(path / f"{k:06d}.npz", allow_pickle=False) as d:
            samples.append(Sample(**row, **{key: d[key] for key in d.files}))
    return samples


def cache_metadata(bag, canonical_lidar, max_per_camera, chunks) -> dict:
    """Bind a cache to its sampling settings and upstream artifact indexes."""
    root = Path(artifact_root().removeprefix("file://")) / "raw" / bag
    sources = [
        root / p
        for p in ("calibration.json", "tracks.parquet", "track_summaries.parquet")
    ]
    for chunk in sorted((root / "chunks").iterdir()):
        if not chunk.is_dir() or (chunks and chunk.name not in chunks):
            continue
        sources.extend(
            chunk / p
            for p in (
                "frame_index.parquet",
                "poses.parquet",
                "lidar_proc_index.parquet",
                "tracklets_2d.parquet",
                "manifest_lidar_preprocessing.json",
                "manifest_perception_2d.json",
            )
        )
    return {
        "sampling_version": 3,  # Shared lens model and full inverse intrinsics.
        "bag": bag,
        "canonical_lidar": canonical_lidar,
        "max_per_camera": max_per_camera,
        "chunks": chunks,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
            if p.is_file()
        },
    }


def parameters(family, p):
    """Optical-frame rotation vector (degrees), translation (metres), K, time.

    ego_T_cam_new = ego_T_cam @ Delta; optical x right, y down, z forward.
    yaw/pitch here mean rotation about optical y/x, respectively.
    """
    rot = np.zeros(3)
    trans = np.zeros(3)
    shift = np.zeros(2)
    scale = 1.0
    dt = 0.0
    if family == "rotation":
        rot[:] = p
    elif family == "yaw":
        rot[1] = p[0]
    elif family == "pitch":
        rot[0] = p[0]
    elif family == "translation":
        trans[:] = p
    elif family == "principal":
        shift[:] = p
    elif family == "focal":
        scale = p[0]
    elif family == "time":
        dt = p[0]
    elif family != "baseline":
        raise ValueError(f"unknown parameter family {family!r}")
    return Rotation.from_rotvec(np.deg2rad(rot)).as_matrix(), trans, shift, scale, dt


def project(sample, cam, family="baseline", p=()):
    R, t, shift, scale, dt = parameters(family, p)
    pts = sample.ego_xyz
    if dt:
        k = int(np.argmin(np.abs(OFFSETS - dt)))
        T = sample.offset_poses[k]
        pts = pts @ T[:3, :3].T + T[:3, 3]
    # A local camera-frame correction: ego_T_cam_new = ego_T_cam @ Delta.
    pts = (pts - cam.ego_T_cam[:3, 3]) @ cam.ego_T_cam[:3, :3]
    pts = (pts - t) @ R
    K = cam.K.copy()
    K[:2, :2] *= scale
    K[:2, 2] += shift
    model = CameraModel(K, cam.model.distortion, cam.model.distortion_model)
    uv, depth, ok = model.project(pts, np.eye(4), (cam.width, cam.height))
    ok &= depth > 0.5
    return uv, ok


def score(sample, cam, family="baseline", p=()):
    uv, ok = project(sample, cam, family, p)
    q = uv - sample.origin
    finite = np.isfinite(q).all(1)
    safe = np.where(finite[:, None], q, 0)
    ui, vi = pixel_indices(safe, sample.mask.shape[1], sample.mask.shape[0])
    incrop = (
        finite
        & (q[:, 0] >= 0)
        & (q[:, 0] < sample.mask.shape[1])
        & (q[:, 1] >= 0)
        & (q[:, 1] < sample.mask.shape[0])
    )
    hits = ok & incrop & sample.mask[vi, ui]
    distance = map_coordinates(
        sample.distance, [safe[:, 1], safe[:, 0]], order=1, mode="constant", cval=100.0
    )
    distance[~ok] = 100.0
    loss = float(np.mean(np.minimum(distance / max(min(sample.extent), 20), 1.0)))
    offset = (
        (uv[ok].mean(0) - sample.centroid) / sample.extent
        if ok.any()
        else np.array([np.nan, np.nan])
    )
    return float(hits.mean()), loss, *offset


def metrics(samples, cam, family="baseline", p=()):
    if not samples:
        return {}
    values = np.array([score(s, cam, family, p) for s in samples])

    def median_finite(v):
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else None

    return dict(
        n=len(samples),
        tracks=len({s.track_id for s in samples}),
        median_inside=float(np.median(values[:, 0])),
        mean_inside=float(np.mean(values[:, 0])),
        outside_loss=float(np.mean(values[:, 1])),
        median_du=median_finite(values[:, 2]),
        median_dv=median_finite(values[:, 3]),
        median_abs_du=median_finite(abs(values[:, 2])),
        median_abs_dv=median_finite(abs(values[:, 3])),
    )


def fit_family(samples, cam, family):
    x0, bounds = FAMILIES[family]
    fit = minimize(
        lambda p: np.mean([score(s, cam, family, p)[1] for s in samples]),
        x0,
        method="Powell",
        bounds=bounds,
        options={"xtol": 0.005, "ftol": 0.0001, "maxiter": 8},
    )
    return fit.x.tolist()


def rotation_gain(samples, cam, p):
    """Equal-weight track means, with bootstrap uncertainty across tracks."""
    by_track = defaultdict(list)
    for s in samples:
        by_track[s.track_id].append(score(s, cam, "rotation", p)[0] - score(s, cam)[0])
    gains = np.array([np.mean(v) for v in by_track.values()])
    rng = np.random.default_rng(0)
    boots = np.mean(rng.choice(gains, (2000, len(gains))), axis=1)
    return {
        "tracks": len(gains),
        "mean_gain": float(gains.mean()),
        "bootstrap_95": np.percentile(boots, [2.5, 97.5]).tolist(),
    }


def run(samples, calib, out):
    result = {}
    for cid in sorted(calib.cameras):
        cam = calib.camera(cid)
        ss = [s for s in samples if s.cam_id == cid and s.cls == "car"]
        tr = [s for s in ss if s.train]
        val = [s for s in ss if not s.train]
        if not tr or not val:
            continue
        entry = {
            "baseline": {"train": metrics(tr, cam), "validation": metrics(val, cam)},
            "fits": {},
        }
        print(cid, "baseline", json.dumps(entry["baseline"]), flush=True)
        for family in FAMILIES:
            p = fit_family(tr, cam, family)
            entry["fits"][family] = {
                "parameters": p,
                "train": metrics(tr, cam, family, p),
                "validation": metrics(val, cam, family, p),
            }
            if family == "rotation":
                entry["fits"][family]["track_gain"] = rotation_gain(val, cam, p)
            print(
                cid,
                family,
                np.round(p, 4),
                "train/val",
                round(entry["fits"][family]["train"]["median_inside"], 3),
                round(entry["fits"][family]["validation"]["median_inside"], 3),
                flush=True,
            )
        time_train = [
            s
            for s in tr
            if s.state == "parked" and s.speed > 1 and np.isfinite(s.offset_poses).all()
        ]
        time_val = [
            s
            for s in val
            if s.state == "parked" and s.speed > 1 and np.isfinite(s.offset_poses).all()
        ]
        if time_train and time_val:
            grid = [(dt, metrics(time_train, cam, "time", [dt])) for dt in OFFSETS]
            dt, tm = min(grid, key=lambda x: x[1]["outside_loss"])
            entry["fits"]["time"] = {
                "parameters": [dt],
                "train": tm,
                "validation": metrics(time_val, cam, "time", [dt]),
                "baseline_train": metrics(time_train, cam),
                "baseline_validation": metrics(time_val, cam),
                "grid": grid,
            }
            print(
                cid,
                "time",
                dt,
                "train/val",
                round(tm["median_inside"], 3),
                round(entry["fits"]["time"]["validation"]["median_inside"], 3),
                flush=True,
            )
        result[cid] = entry
        Path(out).write_text(json.dumps(result, indent=2))
    return result


def candidate_calibration(calib, result):
    """Export only rotation fits with support on independent track IDs.

    Validation is also used to reject regressions, so the exported candidate
    still needs another recording or surveyed correspondences for acceptance.
    Intrinsics, translation and time remain as recorded.
    """
    cameras = dict(calib.cameras)
    selected = {}
    for cid, entry in result.items():
        fit = entry["fits"]["rotation"]
        gain = fit["track_gain"]
        if (
            gain["tracks"] < 20
            or fit["validation"]["n"] < 25
            or gain["mean_gain"] < 0.02
            or gain["bootstrap_95"][0] <= 0
            or fit["validation"]["outside_loss"]
            >= entry["baseline"]["validation"]["outside_loss"]
        ):
            continue
        cam = calib.camera(cid)
        R, *_ = parameters("rotation", fit["parameters"])
        T = cam.ego_T_cam.copy()
        T[:3, :3] = T[:3, :3] @ R
        cameras[cid] = replace(cam, ego_T_cam=T)
        selected[cid] = fit["parameters"]
    return replace(
        calib,
        cameras=cameras,
        version=calib.version + "__projection_audit",
        checks={
            "sanity": "experimental",
            "notes": (
                "Mask-based optical-frame rotation fit on separate track IDs. "
                "Not a surveyed calibration. Raw TF dump retained; camera overrides "
                "differ. Rerun depth fitting and downstream stages before production use."
            ),
        },
    ), selected


def evaluate(samples, calib):
    out = {}
    for cls in ("car", "pedestrian"):
        values = []
        per_camera = {}
        for cid in sorted(calib.cameras):
            ss = [
                s for s in samples if not s.train and s.cls == cls and s.cam_id == cid
            ]
            per_camera[cid] = metrics(ss, calib.camera(cid))
            values.extend(score(s, calib.camera(cid))[0] for s in ss)
        out[cls] = {
            "n": len(values),
            "tracks": len(
                {s.track_id for s in samples if not s.train and s.cls == cls}
            ),
            "per_camera": per_camera,
            "mean_inside": float(np.mean(values)) if values else None,
            "median_inside": float(np.median(values)) if values else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-camera", type=int, default=250)
    ap.add_argument("--canonical-lidar")
    ap.add_argument("--chunks", nargs="+")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--candidate", help="Authored calibration to score without tuning")
    ap.add_argument(
        "--check-min-inside", type=float, help="Fail if validation car median is lower"
    )
    args = ap.parse_args()
    if args.max_per_camera < 1:
        ap.error("--max-per-camera must be positive")
    cache = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata(
        args.bag, args.canonical_lidar, args.max_per_camera, args.chunks
    )
    if (cache / "index.json").exists():
        samples = load_samples(cache, metadata)
    else:
        samples = collect(
            args.bag, args.max_per_camera, args.canonical_lidar, args.chunks
        )
        save_samples(cache, samples, metadata)
    if not samples:
        ap.error("no usable track/mask/point correspondences")
    calib = Calibration.load(args.bag)
    if args.candidate:
        candidate = Calibration.from_dict(json.loads(Path(args.candidate).read_text()))
    elif args.evaluate_only:
        candidate = calib
    else:
        result = run(samples, calib, out / "sweep.json")
        candidate, selected = candidate_calibration(calib, result)
        (out / "calibration_candidate.json").write_text(
            json.dumps(candidate.to_dict(), indent=2) + "\n"
        )
        (out / "selected_rotations.json").write_text(
            json.dumps(selected, indent=2) + "\n"
        )
    report = {
        "inputs": metadata,
        "baseline": evaluate(samples, calib),
        "candidate": evaluate(samples, candidate),
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    cls: {k: v for k, v in stats.items() if k != "per_camera"}
                    for cls, stats in report[name].items()
                }
                for name in ("baseline", "candidate")
            },
            indent=2,
        )
    )
    median = report["candidate"]["car"]["median_inside"]
    if args.check_min_inside is not None and (
        median is None or median < args.check_min_inside
    ):
        print(f"FAIL: validation car containment {median} < {args.check_min_inside}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
