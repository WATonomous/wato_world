"""Import boxes drawn in SUSTechPOINTS as ground truth (gt_boxes.parquet).

Reads every scene export_sustechpoints.py wrote for the bag
(<bag>/labelling/sustechpoints/*/): each label/<frame>.json is in the ego frame
at that frame's reference time, which frames.json records with the frame's
world_T_ego, so every box goes back to the world frame. psr.scale is
(length, width, height) and psr.rotation.z the yaw. An object is one obj_id
within a scene (instance_id "<chunk_id>-<obj_id>"), and its parked / moving
state comes from its boxes over time (wato_common.eval.motion).

    python /ws/src/ingest/scripts/import_sustechpoints.py <bag_id>

Replaces the bag's gt_boxes.parquet.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from wato_common.artifacts import gt_boxes
from wato_common.eval.motion import motion_state
from export_sustechpoints import scene_dir  # noqa: E402 — sibling script

SOURCE = "sustechpoints"

# SUSTechPOINTS obj_type → the pipeline's class; anything else is kept with
# cls None so evaluation ignores it.
CLASS_BY_TYPE = {
    "Car": "car",
    "Van": "car",
    "PoliceCar": "car",
    "TourCar": "car",
    "Truck": "truck",
    "LongVehicle": "truck",
    "Bus": "bus",
    "Motorcycle": "motorcycle",
    "MotorcyleRider": "motorcycle",
    "Bicycle": "bicycle",
    "BicycleRider": "bicycle",
    "Pedestrian": "pedestrian",
    "Child": "pedestrian",
    "RoadWorker": "pedestrian",
}


def scene_boxes(path: str) -> list[dict]:
    with open(os.path.join(path, "frames.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    chunk_id = meta["chunk_id"]
    out = []
    for name, f in sorted(meta["frames"].items()):
        label = os.path.join(path, "label", f"{name}.json")
        if not os.path.exists(label):
            continue
        world_T_ego = np.array(f["world_T_ego"]).reshape(4, 4)
        yaw_ego_to_world = math.atan2(world_T_ego[1, 0], world_T_ego[0, 0])
        with open(label, encoding="utf-8") as fh:
            objs = json.load(fh)
        for o in objs:
            p, s, r = o["psr"]["position"], o["psr"]["scale"], o["psr"]["rotation"]
            c = world_T_ego @ np.array([p["x"], p["y"], p["z"], 1.0])
            heading = math.remainder(r["z"] + yaw_ego_to_world, 2.0 * math.pi)
            out.append(
                dict(
                    source=SOURCE,
                    timestamp_ns=int(f["reference_timestamp_ns"]),
                    instance_id=f"{chunk_id}-{o['obj_id']}",
                    cx=float(c[0]),
                    cy=float(c[1]),
                    cz=float(c[2]),
                    w=float(s["y"]),
                    l=float(s["x"]),
                    h=float(s["z"]),
                    heading=heading,
                    cls=CLASS_BY_TYPE.get(o["obj_type"]),
                    source_class=o["obj_type"],
                )
            )
    return out


def with_motion_states(boxes: list[dict]) -> list[gt_boxes.GtBox]:
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for b in boxes:
        by_inst[b["instance_id"]].append(b)
    state = {
        iid: motion_state(
            np.array([b["timestamp_ns"] for b in bs]),
            np.array([[b["cx"], b["cy"]] for b in bs]),
        )
        for iid, bs in by_inst.items()
    }
    return [gt_boxes.GtBox(**b, motion_state=state[b["instance_id"]]) for b in boxes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag_id")
    args = ap.parse_args()

    scenes = sorted(
        glob.glob(os.path.join(os.path.dirname(scene_dir(args.bag_id, "x")), "*"))
    )
    boxes = [b for s in scenes for b in scene_boxes(s)]
    if not boxes:
        raise SystemExit(
            f"no labels under {os.path.dirname(scene_dir(args.bag_id, 'x'))}"
        )
    gt = with_motion_states(boxes)
    gt_boxes.write(args.bag_id, gt)
    states = Counter(
        (b.cls, b.motion_state) for b in {b.instance_id: b for b in gt}.values()
    )
    print(f"{args.bag_id}: {len(gt)} boxes from {len(scenes)} scene(s)")
    for (cls, st), n in sorted(
        states.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
    ):
        print(f"  {str(cls):<11} {str(st):<7} {n}")
    unknown = Counter(b.source_class for b in gt if b.cls is None)
    if unknown:
        print(f"  ignored types: {dict(unknown)}")


if __name__ == "__main__":
    main()
