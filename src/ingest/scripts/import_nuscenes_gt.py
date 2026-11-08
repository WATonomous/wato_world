"""Import nuScenes keyframe annotations from a converted bag as ground truth.

The nuScenes → ROS 2 converter publishes every sample's annotations as one
visualization_msgs/MarkerArray: a CUBE per box in the `map` frame (the frame
ingest's nuScenes profile takes poses in), stamped at the sample's time, with
`id` naming the instance across samples, `scale` = (length, width, height) and
`color` = the devkit's category colour. The category name itself is not in the
message, so it is read back from that colour; an unknown colour stops the
import rather than guessing.

    # in the ingest_dev container:
    python /ws/src/ingest/scripts/import_nuscenes_gt.py \
        /data/bags/NuScenes-v1.0-mini-scene-0061_0.mcap \
        --bag NuScenes_v1_0_mini_scene_0061_0_mcap

Writes gt_boxes.parquet (wato_common.artifacts.gt_boxes) with each instance's
parked / moving state judged by wato_common.eval.motion.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np

from wato_common.artifacts import gt_boxes
from wato_common.eval.motion import motion_state
from wato_common.io.rosbag_reader import messages

SOURCE = "nuscenes_markers"
CUBE = 1  # visualization_msgs/Marker.CUBE

# nuscenes-devkit python-sdk/nuscenes/utils/color_map.py (get_colormap), RGB.
CATEGORY_BY_RGB = {
    (0, 0, 0): "noise",
    (70, 130, 180): "animal",
    (0, 0, 230): "human.pedestrian.adult",
    (135, 206, 235): "human.pedestrian.child",
    (100, 149, 237): "human.pedestrian.construction_worker",
    (219, 112, 147): "human.pedestrian.personal_mobility",
    (0, 0, 128): "human.pedestrian.police_officer",
    (240, 128, 128): "human.pedestrian.stroller",
    (138, 43, 226): "human.pedestrian.wheelchair",
    (112, 128, 144): "movable_object.barrier",
    (210, 105, 30): "movable_object.debris",
    (105, 105, 105): "movable_object.pushable_pullable",
    (47, 79, 79): "movable_object.trafficcone",
    (188, 143, 143): "static_object.bicycle_rack",
    (220, 20, 60): "vehicle.bicycle",
    (255, 127, 80): "vehicle.bus.bendy",
    (255, 69, 0): "vehicle.bus.rigid",
    (255, 158, 0): "vehicle.car",
    (233, 150, 70): "vehicle.construction",
    (255, 83, 0): "vehicle.emergency.ambulance",
    (255, 215, 0): "vehicle.emergency.police",
    (255, 61, 99): "vehicle.motorcycle",
    (255, 140, 0): "vehicle.trailer",
    (255, 99, 71): "vehicle.truck",
}

# nuScenes category → the pipeline's class (perception_2d's taxonomy), as the
# nuScenes detection benchmark groups them. Anything else maps to None: kept,
# so evaluation can ignore it instead of counting it as a miss.
CLASS_BY_CATEGORY = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
}


def _yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def _category(marker) -> str:
    rgb = tuple(
        int(round(c * 255)) for c in (marker.color.r, marker.color.g, marker.color.b)
    )
    try:
        return CATEGORY_BY_RGB[rgb]
    except KeyError:
        raise ValueError(
            f"marker {marker.id}: colour {rgb} is no nuScenes category colour; "
            "the converter's colour map differs from the devkit's"
        ) from None


def read_boxes(bag_path: str, topic: str, frame: str) -> list[dict]:
    boxes = []
    with messages(bag_path, topics=[topic]) as it:
        for _, msg, _ in it:
            for m in msg.markers:
                if m.type != CUBE:
                    raise ValueError(f"marker {m.id}: type {m.type}, expected CUBE")
                if m.header.frame_id != frame:
                    raise ValueError(
                        f"marker {m.id}: frame {m.header.frame_id!r}, expected {frame!r}"
                    )
                category = _category(m)
                boxes.append(
                    dict(
                        source=SOURCE,
                        timestamp_ns=m.header.stamp.sec * 1_000_000_000
                        + m.header.stamp.nanosec,
                        instance_id=str(m.id),
                        cx=m.pose.position.x,
                        cy=m.pose.position.y,
                        cz=m.pose.position.z,
                        w=m.scale.y,
                        l=m.scale.x,
                        h=m.scale.z,
                        heading=_yaw(m.pose.orientation),
                        cls=CLASS_BY_CATEGORY.get(category),
                        source_class=category,
                    )
                )
    return boxes


def with_motion_states(boxes: list[dict]) -> list[gt_boxes.GtBox]:
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for b in boxes:
        by_instance[b["instance_id"]].append(b)
    state = {
        iid: motion_state(
            np.array([b["timestamp_ns"] for b in bs]),
            np.array([[b["cx"], b["cy"]] for b in bs]),
        )
        for iid, bs in by_instance.items()
    }
    return [gt_boxes.GtBox(**b, motion_state=state[b["instance_id"]]) for b in boxes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag_path")
    ap.add_argument(
        "--bag", dest="bag_id", required=True, help="bag_id ingest registered"
    )
    ap.add_argument("--topic", default="/markers/annotations")
    ap.add_argument("--frame", default="map", help="the frame ingest's poses are in")
    args = ap.parse_args()

    boxes = with_motion_states(read_boxes(args.bag_path, args.topic, args.frame))
    gt_boxes.write(args.bag_id, boxes)
    n_inst = len({b.instance_id for b in boxes})
    by_state: dict[object, set[str]] = defaultdict(set)
    for b in boxes:
        by_state[(b.cls, b.motion_state)].add(b.instance_id)
    print(f"{args.bag_id}: {len(boxes)} boxes, {n_inst} instances")
    for (cls, st), ids in sorted(
        by_state.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
    ):
        print(f"  {str(cls):<11} {str(st):<7} {len(ids)}")


if __name__ == "__main__":
    main()
