"""Export one chunk as a SUSTechPOINTS scene, for drawing ground-truth boxes.

SUSTechPOINTS (github.com/naurril/SUSTechPOINTS) reads a scene directory:

  lidar/<frame>.pcd              points in one fixed sensor frame per frame
  camera/<cam>/<frame>.jpg       images, one per camera per frame
  calib/camera/<cam>.json        {"extrinsic": 4x4 lidar→camera, "intrinsic": 3x3}
  label/<frame>.json             written by the tool: [{obj_id, obj_type, psr}]

Its calibration is fixed per scene, so each frame's deskewed world points are
written in the ego frame at that frame's reference time, and the extrinsic is
the camera's (static) cam_T_ego. Images are the ones ingest paired with the
frame's reference sweep; they were taken up to ~44 ms from it, so a box can sit
slightly off a moving object in the image — draw boxes on the points. A
frames.json sidecar records each frame's sweep, time and world_T_ego, which
import_sustechpoints.py needs to put the labels back into the world frame.

    # in any component's dev container, or on a host with src/common:
    python /ws/src/ingest/scripts/export_sustechpoints.py <bag_id> --chunk 0005 \
        --canonical-lidar lidar_cc [--every 10]

Writes <bag>/labelling/sustechpoints/<bag>_<chunk>/ under the artifact root;
point SUSTechPOINTS' data/ directory at it (a symlink is enough). Keep the
same obj_id for one object in every frame: its parked / moving state is judged
from its boxes over time. Label chunks that don't overlap (not neighbours), or
an object in the shared span is labelled twice.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from wato_common.artifact_store import bag_root, local_path
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.frame_index import FrameIndex
from wato_common.artifacts.lidar_sweeps import ProcessedSweeps
from wato_common.geometry.transforms import invert_se3
from wato_common.pose_lookup import PoseLookup


def scene_dir(bag_id: str, chunk_id: str) -> str:
    return os.path.join(
        local_path(bag_root(bag_id)),
        "labelling",
        "sustechpoints",
        f"{bag_id}_{chunk_id}",
    )


def write_pcd(path: str, xyz: np.ndarray, intensity: np.ndarray) -> None:
    pts = np.empty(
        xyz.shape[0], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4")]
    )
    pts["x"], pts["y"], pts["z"], pts["i"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], intensity
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
        "FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {pts.shape[0]}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {pts.shape[0]}\nDATA binary\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(pts.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag_id")
    ap.add_argument("--chunk", required=True)
    ap.add_argument("--canonical-lidar", default=None)
    ap.add_argument(
        "--every", type=int, default=10, help="every Nth frame (10 = 2 Hz at 20 Hz)"
    )
    args = ap.parse_args()

    out = scene_dir(args.bag_id, args.chunk)
    sweeps = ProcessedSweeps.load(args.bag_id, args.chunk)
    frames = sweeps.frames(args.canonical_lidar)[:: args.every]
    poses = PoseLookup.load(args.bag_id, args.chunk)
    cams_by_sweep = FrameIndex.load(args.bag_id, args.chunk)
    calib = Calibration.load(args.bag_id)

    os.makedirs(os.path.join(out, "lidar"), exist_ok=True)
    os.makedirs(os.path.join(out, "calib", "camera"), exist_ok=True)
    os.makedirs(os.path.join(out, "label"), exist_ok=True)
    for cam_id in sorted(calib.cameras):
        cam = calib.camera(cam_id)
        with open(os.path.join(out, "calib", "camera", f"{cam_id}.json"), "w") as fh:
            json.dump(
                dict(
                    extrinsic=invert_se3(cam.ego_T_cam).reshape(-1).tolist(),
                    intrinsic=np.asarray(cam.K).reshape(-1).tolist(),
                ),
                fh,
                indent=2,
            )

    sidecar = {}
    for f in frames:
        pose = poses.at(f.reference_timestamp_ns)
        if not pose.valid:
            continue
        name = f"{f.reference_sweep_id:06d}"
        ego_T_world = invert_se3(pose.world_T_ego)
        xyz, inten = [], []
        for sid in f.sweep_ids:
            w = sweeps.world(sid)
            xyz.append(w.xyz)
            inten.append(
                w.intensity.astype(np.float32)
                if w.intensity is not None
                else np.zeros(w.n_points, np.float32)
            )
        p = np.concatenate(xyz)
        p_ego = p @ ego_T_world[:3, :3].T + ego_T_world[:3, 3]
        write_pcd(
            os.path.join(out, "lidar", f"{name}.pcd"), p_ego, np.concatenate(inten)
        )
        for cf in cams_by_sweep.for_sweep(f.reference_sweep_id):
            dst = os.path.join(out, "camera", cf.cam_id)
            os.makedirs(dst, exist_ok=True)
            ext = os.path.splitext(cf.image_path)[1] or ".jpg"
            shutil.copyfile(cf.image_path, os.path.join(dst, f"{name}{ext}"))
        sidecar[name] = dict(
            reference_sweep_id=f.reference_sweep_id,
            sweep_ids=list(f.sweep_ids),
            reference_timestamp_ns=f.reference_timestamp_ns,
            world_T_ego=pose.world_T_ego.reshape(-1).tolist(),
        )
    with open(os.path.join(out, "frames.json"), "w") as fh:
        json.dump(
            dict(bag_id=args.bag_id, chunk_id=args.chunk, frames=sidecar), fh, indent=1
        )
    print(f"wrote {len(sidecar)} frames to {out}")


if __name__ == "__main__":
    main()
