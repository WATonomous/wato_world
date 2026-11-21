"""Pipeline outputs merged into a copy of the source bag, for debugging.

`debug-bag` writes a bag's boxes as ROS topics next to the original sensor
topics, so Foxglove or RViz shows each box over the LiDAR returns and images
it came from. One visualization_msgs/MarkerArray topic per layer that exists:

  /wato_world/proposals/<provenance>   proposal_generation's rows, one topic
                                       per provenance (lidar_mos, lifted,
                                       fused, ...); coloured by class
  /wato_world/tracks                   tracking's boxes as proposed, coloured
                                       per track id: a fragmented object
                                       changes colour
  /wato_world/refined_labels           label_refinement's boxes, coloured by
                                       class, with an arrow where the heading
                                       has a front

One message per frame and layer: a DELETEALL per namespace, then a
translucent CUBE ("boxes"), a TEXT_VIEW_FACING label ("labels") and, for a
directed heading, an ARROW ("headings") per box, the shape of wato_monorepo's
world_objects_markers. A layer with no box in a frame still gets a message,
which clears the previous frame's.

Opt-in camera layers, one sensor_msgs/CompressedImage (JPEG) per camera
image ingest decoded, drawn by wato_ingest.debug_images:

  /wato_world/masks/<cam>   (--masks) the image with perception_2d's masklets
                            tinted, outlined and labelled
  /wato_world/depth/<cam>   (--depth) perception_2d's metric depth, coloured

<cam> is the cam_id in lower case. Each carries its source image's header
stamp, log time and frame_id (camera_frames.parquet, calibration.json), so it
replays in lockstep with the camera topic and Foxglove pairs it with the same
CameraInfo and TF. Every image gets one, empty ones included, so a stale
mask never stays on screen.

Boxes are in the ego frame (calibration.json's ego_frame), each frame at its
reference time with the pose interpolated there (PoseLookup); log time =
header stamp = that time, so the boxes play in step with the sensor data. Not
the pipeline's world frame: the bag has no trustworthy TF for it (WATO's /tf
is stamped with wall-clock time by eidos_transform, nuScenes' is 2 Hz), and a
second parent for the ego frame would fight the bag's own tree. View with the
ego frame as the display frame. A frame with no valid pose (an untrusted
stretch) is published empty, so no box from an earlier frame stays on screen.

Every instant comes from the chunk that owns it (wato_common.chunk_spans), as
every eval scores it: neighbouring chunks share 6 s and both have boxes there
(tracks' and refined labels' owned rows are the same instants); an image by
its header stamp, so a masklet id or colour changes at the seam. With a chunk
named, that chunk's rows and images over its whole decoded window, and the
source cut to that window (static topics whole, as trim-bag does).

Copies use the pinned mcap CLI, as trim-bag does: rosbag2's writer re-derives
schemas from this image's message packages and would empty the source's
custom types. Only the new topics go through rosbag2:

    1. rosbag2 writer -> overlay (the new topics only)
    2. mcap filter <overlay> --exclude-metadata (the output keeps the source's)
    3. mcap filter <source> -> window part + static part (chunk only)
    4. mcap merge <source or parts> <overlay> -> output
"""

from __future__ import annotations

import heapq
import json
import math
import multiprocessing
import shutil
import subprocess
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from wato_common import chunk_spans
from wato_common.artifact_store import (
    bag_meta_path,
    camera_frames_path,
    chunks_index_path,
    local_path,
)
from wato_common.artifacts import depth, masklets, proposals, refined_labels, tracks
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts.proposals import Proposal
from wato_common.artifacts.refined_labels import RefinedBox
from wato_common.artifacts.tracks import TrackFrame, TrackSummary
from wato_common.io.parquet_io import read_rows
from wato_common.pose_lookup import PoseLookup
from wato_common.schemas import Box3D
from wato_ingest import debug_images
from wato_ingest.debug_images import ImageFrame, id_rgb
from wato_ingest.inputs.trim import resolve_mcap_file, topic_regex

TOPIC_ROOT = "/wato_world/"
TRACKS = "tracks"
REFINED = "refined_labels"
MARKER_ARRAY = "visualization_msgs/msg/MarkerArray"
COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
JPEG_FORMAT = "rgb8; jpeg compressed bgr8"  # image_transport's, as WATO records
NAMESPACES = ("boxes", "labels", "headings")
BOX_ALPHA = 0.35
LABEL_RGBA = (1.0, 1.0, 1.0, 0.8)
LABEL_HEIGHT_M = 0.4
ARROW_BEYOND_M = 1.0  # a heading arrow reaches this far past the box's front
# Class colours (RGB); any other class, or none, is grey.
CLASS_RGB = {
    "car": (0.2, 0.6, 1.0),
    "truck": (1.0, 0.6, 0.1),
    "bus": (0.8, 0.3, 0.9),
    "pedestrian": (1.0, 0.25, 0.25),
    "bicycle": (0.2, 0.9, 0.4),
    "motorcycle": (0.95, 0.9, 0.2),
}
UNCLASSED_RGB = (0.7, 0.7, 0.7)


class DebugBagError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# What to draw, as ego-frame boxes (no ROS).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Drawn:
    """A world box to draw in one layer at one frame."""

    layer: str
    sweep_id: int  # the frame's reference sweep
    t_ns: int  # its reference timestamp
    box: Box3D
    text: str
    rgb: tuple[float, float, float]
    directed: bool = False  # heading has a front: draw an arrow
    key: str = ""  # orders a frame's boxes, so marker ids are stable


@dataclass(frozen=True)
class EgoBox:
    """One box in the ego frame at its frame's time."""

    centre: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    size: tuple[float, float, float]  # l, w, h along the box's x, y, z
    rgba: tuple[float, float, float, float]
    text: str
    directed: bool = False


@dataclass(frozen=True)
class DebugFrame:
    """One frame: its time and its boxes per layer."""

    chunk_id: str
    sweep_id: int  # the frame's reference sweep
    t_ns: int  # its reference timestamp
    boxes: dict[str, tuple[EgoBox, ...]]  # layer -> boxes; empty without a pose
    has_pose: bool


def topic(layer: str) -> str:
    return TOPIC_ROOT + layer


def proposal_layer(provenance: str) -> str:
    return f"proposals/{provenance}"


def class_rgb(cls: Optional[str]) -> tuple[float, float, float]:
    return CLASS_RGB.get(cls or "", UNCLASSED_RGB)


def image_topic(layer: str, cam_id: str) -> str:
    return f"{TOPIC_ROOT}{layer}/{cam_id.lower()}"


def track_rgb(track_id: str) -> tuple[float, float, float]:
    """A colour per track id, the same on every run; neighbouring ids differ."""
    return id_rgb(track_id)


def box_of(r) -> Box3D:
    return Box3D(cx=r.cx, cy=r.cy, cz=r.cz, w=r.w, l=r.l, h=r.h, heading=r.heading)


def ego_box(d: Drawn, ego_T_world: np.ndarray) -> EgoBox:
    """d's world box in the ego frame. The box is gravity-aligned in the world
    (yaw only), so in the ego frame it takes the ego's roll and pitch too."""
    b = d.box
    c = ego_T_world @ np.array([b.cx, b.cy, b.cz, 1.0])
    yaw = Rotation.from_euler("z", b.heading).as_matrix()
    q = Rotation.from_matrix(ego_T_world[:3, :3] @ yaw).as_quat()  # x, y, z, w
    return EgoBox(
        centre=(float(c[0]), float(c[1]), float(c[2])),
        quat_xyzw=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
        size=(b.l, b.w, b.h),
        rgba=(*d.rgb, BOX_ALPHA),
        text=d.text,
        directed=d.directed,
    )


def label(p: Proposal, by_id: dict[str, Proposal]) -> str:
    """A proposal's class and score, then what backs the box (two lines)."""
    head = f"{p.cls or '?'} {p.score:.2f}"
    n = f"n={p.lidar_point_count}" if p.lidar_point_count is not None else ""
    if p.provenance == proposals.FUSED:
        members = sorted(
            {by_id[m].provenance for m in p.member_proposal_ids if m in by_id}
        )
        detail = f"[{'+'.join(members)}] {n}"
    elif p.provenance == "lifted":
        conf = f"conf={p.lift_confidence:.2f}" if p.lift_confidence is not None else ""
        seeds = (
            f"seeds={p.lift_seed_point_count}"
            if p.lift_seed_point_count is not None
            else ""
        )
        detail = " ".join(x for x in (conf, seeds, n) if x)
    elif p.provenance == "lidar_mos":
        motion = f"motion={p.motion_score:.2f}" if p.motion_score is not None else ""
        detail = " ".join(x for x in (motion, n) if x)
    else:
        detail = n
    return f"{head}\n{detail.strip()}" if detail.strip() else head


def proposal_items(rows: list[Proposal]) -> list[Drawn]:
    by_id = {p.proposal_id: p for p in rows}
    return [
        Drawn(
            layer=proposal_layer(p.provenance),
            sweep_id=p.sweep_id,
            t_ns=p.reference_timestamp_ns,
            box=box_of(p),
            text=label(p, by_id),
            rgb=class_rgb(p.cls),
            key=p.proposal_id,
        )
        for p in rows
    ]


def track_items(rows: list[TrackFrame], summaries: list[TrackSummary]) -> list[Drawn]:
    """Each track frame's box as proposed, labelled with its track."""
    state = {s.track_id: s.motion_state for s in summaries}
    return [
        Drawn(
            layer=TRACKS,
            sweep_id=f.sweep_id,
            t_ns=f.reference_timestamp_ns,
            box=box_of(f),
            text=f"{f.track_id} {f.cls or '?'}\n{state.get(f.track_id, '?')}",
            rgb=track_rgb(f.track_id),
            key=f.track_id,
        )
        for f in rows
    ]


def refined_items(rows: list[RefinedBox]) -> list[Drawn]:
    return [
        Drawn(
            layer=REFINED,
            sweep_id=r.sweep_id,
            t_ns=r.reference_timestamp_ns,
            box=box_of(r),
            text=(
                f"{r.track_id} {r.cls or '?'} {r.confidence:.2f}\n"
                f"{r.motion_state} {r.size_source}"
            ),
            rgb=class_rgb(r.cls),
            directed=r.heading_directed,
            key=r.track_id,
        )
        for r in rows
    ]


def chunk_frames(
    chunk_id: str,
    items: list[Drawn],
    poses: PoseLookup,
    window: tuple[float, float],
    layers: list[str],
) -> list[DebugFrame]:
    """The chunk's frames with a reference time in window ([lo, hi)), in time
    order, each with a (possibly empty) box list per layer."""
    by_frame: dict[tuple[int, int], list[Drawn]] = defaultdict(list)
    for d in items:
        if window[0] <= d.t_ns < window[1]:
            by_frame[(d.t_ns, d.sweep_id)].append(d)
    out = []
    for (t_ns, sweep_id), ds in sorted(by_frame.items()):
        pose = poses.at(t_ns)
        boxes: dict[str, list[EgoBox]] = {layer: [] for layer in layers}
        if pose.valid:
            ego_T_world = np.linalg.inv(pose.world_T_ego)
            for d in sorted(ds, key=lambda d: (d.layer, d.key)):
                boxes[d.layer].append(ego_box(d, ego_T_world))
        out.append(
            DebugFrame(
                chunk_id=chunk_id,
                sweep_id=sweep_id,
                t_ns=t_ns,
                boxes={k: tuple(v) for k, v in boxes.items()},
                has_pose=pose.valid,
            )
        )
    return out


def chunk_windows(
    bag_id: str, chunk_id: Optional[str] = None
) -> dict[str, tuple[float, float]]:
    """chunk_id -> the [lo, hi) of the instants drawn from it, in time order:
    every chunk's owned window, or chunk_id's whole timeline."""
    index = sorted(read_rows(chunks_index_path(bag_id)), key=lambda r: r["t_start_ns"])
    if not index:
        raise DebugBagError(f"bag {bag_id!r}: no chunks (run ingest)")
    ids = [r["chunk_id"] for r in index]
    if chunk_id is not None:
        if chunk_id not in ids:
            raise DebugBagError(
                f"bag {bag_id!r} has no chunk {chunk_id!r}; chunks: {ids}"
            )
        return {chunk_id: (-math.inf, math.inf)}
    return chunk_spans.owned_windows(index)


def load_frames(bag_id: str, chunk_id: Optional[str] = None) -> list[DebugFrame]:
    """The bag's frames with every layer that exists (proposals, tracks,
    refined labels), each instant from the chunk that owns it, or every frame
    of chunk_id."""
    windows = chunk_windows(bag_id, chunk_id)
    wanted = list(windows)
    missing = [c for c in wanted if not proposals.exists(bag_id, c)]
    if missing:
        raise DebugBagError(
            f"bag {bag_id!r}: no proposals for chunks {missing} (run "
            "proposal_generation, or name a chunk that has them)"
        )
    items: dict[str, list[Drawn]] = {
        c: proposal_items(proposals.read(bag_id, c)) for c in wanted
    }
    if tracks.exists(bag_id):
        rows, summaries = tracks.read(bag_id)
        for d, f in zip(track_items(rows, summaries), rows):
            if f.chunk_id in items:
                items[f.chunk_id].append(d)
    if refined_labels.exists(bag_id):
        rows = refined_labels.read(bag_id)
        for d, r in zip(refined_items(rows), rows):
            if r.chunk_id in items:
                items[r.chunk_id].append(d)
    present = {d.layer for ds in items.values() for d in ds}
    layers = [
        proposal_layer(p) for p in proposals.PROVENANCES if proposal_layer(p) in present
    ] + [layer for layer in (TRACKS, REFINED) if layer in present]
    out: list[DebugFrame] = []
    for c in wanted:
        out += chunk_frames(c, items[c], PoseLookup.load(bag_id, c), windows[c], layers)
    return out


def chunk_images(
    bag_id: str,
    chunk_id: str,
    camera_rows: list[dict],
    window: tuple[float, float],
    frame_ids: dict[str, str],
    index: Optional[masklets.MaskletIndex] = None,
) -> list[ImageFrame]:
    """The chunk's decoded images (camera_frames.parquet rows) with a header
    stamp in window ([lo, hi)), each with the masklets index has in it."""
    out = []
    for r in camera_rows:
        if not r["valid"] or not window[0] <= r["header_timestamp_ns"] < window[1]:
            continue
        cam, seq = str(r["cam_id"]), int(r["camera_seq"])
        if cam not in frame_ids:
            raise DebugBagError(f"calibration.json has no camera {cam!r}")
        out.append(
            ImageFrame(
                bag_id=bag_id,
                chunk_id=chunk_id,
                cam_id=cam,
                camera_seq=seq,
                image_path=local_path(r["image_path"]),
                header_ns=int(r["header_timestamp_ns"]),
                log_ns=int(r["record_timestamp_ns"]),
                frame_id=frame_ids[cam],
                masklets=tuple(index.at(cam, seq)) if index is not None else (),
            )
        )
    return out


def load_images(
    bag_id: str, chunk_id: Optional[str], layers: tuple[str, ...]
) -> list[ImageFrame]:
    """Every decoded camera image to draw layers on, each instant from the
    chunk that owns it (or every image of chunk_id), in log-time order."""
    calib = Calibration.load(bag_id)
    frame_ids = {c: cam.frame_id for c, cam in calib.cameras.items()}
    empty = sorted(c for c, f in frame_ids.items() if not f)
    if empty:
        raise DebugBagError(f"calibration.json names no frame_id for cameras {empty}")
    out: list[ImageFrame] = []
    for c, window in chunk_windows(bag_id, chunk_id).items():
        index = None
        if debug_images.MASKS in layers:
            if not masklets.exists(bag_id, c):
                raise DebugBagError(
                    f"bag {bag_id!r} chunk {c}: no masklets (run perception_2d)"
                )
            index = masklets.MaskletIndex.load(bag_id, c)
        rows = read_rows(camera_frames_path(bag_id, c))
        images = chunk_images(bag_id, c, rows, window, frame_ids, index)
        if debug_images.DEPTH in layers and not any(
            depth.exists(bag_id, c, f.cam_id, f.camera_seq) for f in images
        ):
            raise DebugBagError(
                f"bag {bag_id!r} chunk {c}: no depth (run perception_2d)"
            )
        out += images
    return sorted(out, key=lambda f: (f.log_ns, f.cam_id))


# ---------------------------------------------------------------------------
# ROS messages and the overlay bag (rosbag2).
# ---------------------------------------------------------------------------


def marker_array(boxes: tuple[EgoBox, ...], frame_id: str, t_ns: int):
    """A DELETEALL per namespace, then each box's CUBE, label and, for a
    directed heading, arrow."""
    from visualization_msgs.msg import Marker, MarkerArray

    sec, nanosec = divmod(int(t_ns), 1_000_000_000)

    def marker(ns: str, mid: int, kind: int) -> Marker:
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp.sec, m.header.stamp.nanosec = sec, nanosec
        m.ns, m.id, m.type = ns, mid, kind
        return m

    def place(m: Marker, b: EgoBox) -> None:
        p, q = m.pose.position, m.pose.orientation
        p.x, p.y, p.z = b.centre
        q.x, q.y, q.z, q.w = b.quat_xyzw

    out = MarkerArray()
    for ns in NAMESPACES:
        m = marker(ns, 0, Marker.CUBE)
        m.action = Marker.DELETEALL
        out.markers.append(m)
    for i, b in enumerate(boxes):
        cube = marker("boxes", i, Marker.CUBE)
        cube.action = Marker.ADD
        place(cube, b)
        cube.scale.x, cube.scale.y, cube.scale.z = (max(s, 0.01) for s in b.size)
        cube.color.r, cube.color.g, cube.color.b, cube.color.a = b.rgba
        out.markers.append(cube)

        text = marker("labels", i, Marker.TEXT_VIEW_FACING)
        text.action = Marker.ADD
        p = text.pose.position
        p.x, p.y = b.centre[0], b.centre[1]
        p.z = b.centre[2] + b.size[2] / 2.0 + LABEL_HEIGHT_M
        text.pose.orientation.w = 1.0
        text.scale.z = LABEL_HEIGHT_M
        text.color.r, text.color.g, text.color.b, text.color.a = LABEL_RGBA
        text.text = b.text
        out.markers.append(text)

        if b.directed:
            # From the centre along the box's +x (its front) and past it.
            arrow = marker("headings", i, Marker.ARROW)
            arrow.action = Marker.ADD
            place(arrow, b)
            arrow.scale.x = b.size[0] / 2.0 + ARROW_BEYOND_M  # length
            arrow.scale.y, arrow.scale.z = 0.15, 0.3  # shaft, head diameters
            arrow.color.r, arrow.color.g, arrow.color.b = b.rgba[:3]
            arrow.color.a = 0.9
            out.markers.append(arrow)
    return out


def compressed_image(f: ImageFrame, data: bytes):
    """A JPEG with f's source header (stamp and frame_id)."""
    from sensor_msgs.msg import CompressedImage

    m = CompressedImage()
    m.header.frame_id = f.frame_id
    m.header.stamp.sec, m.header.stamp.nanosec = divmod(f.header_ns, 1_000_000_000)
    m.format = JPEG_FORMAT
    m.data = data
    return m


def rendered(
    images: list[ImageFrame], layers: tuple[str, ...], pool=None
) -> Iterator[tuple[ImageFrame, dict[str, bytes]]]:
    """Each image with its drawn layers, in the images' order."""
    draw = partial(debug_images.render, layers=layers)
    results = pool.imap(draw, images, chunksize=4) if pool else map(draw, images)
    return zip(images, results)


def write_overlay(
    path: Path,
    frames: list[DebugFrame],
    frame_id: str,
    images: list[ImageFrame] = (),
    image_layers: tuple[str, ...] = (),
    workers: int = 1,
) -> Counter:
    """A bag at path (a rosbag2 directory) with one MarkerArray per frame and
    layer, log time = header stamp = the frame's time, and one
    CompressedImage per image and image layer, at its source's log time and
    header stamp. Images are drawn in `workers` processes. Returns messages
    per topic."""
    layers = sorted({layer for f in frames for layer in f.boxes})
    topics = {topic(layer): MARKER_ARRAY for layer in layers}
    cams = sorted({f.cam_id for f in images})
    for layer in image_layers:
        topics |= {image_topic(layer, c): COMPRESSED_IMAGE for c in cams}

    def boxes():
        for f in sorted(frames, key=lambda f: (f.t_ns, f.chunk_id)):
            for layer in layers:
                yield (
                    f.t_ns,
                    topic(layer),
                    marker_array(f.boxes.get(layer, ()), frame_id, f.t_ns),
                )

    def pictures(pool):
        if not image_layers:
            return
        order = sorted(images, key=lambda f: (f.log_ns, f.cam_id))
        for f, drawn in rendered(order, image_layers, pool):
            for layer in image_layers:
                msg = compressed_image(f, drawn[layer])
                yield f.log_ns, image_topic(layer, f.cam_id), msg

    with ExitStack() as stack:
        pool = None
        if images and image_layers and workers > 1:
            # Spawned, not forked: pyarrow's threads are already running, and
            # a forked child can inherit a lock one of them holds.
            ctx = multiprocessing.get_context("spawn")
            pool = stack.enter_context(ctx.Pool(workers))
        import rosbag2_py
        from rclpy.serialization import serialize_message

        w = rosbag2_py.SequentialWriter()
        w.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        for i, (name, kind) in enumerate(topics.items()):
            w.create_topic(rosbag2_py.TopicMetadata(i, name, kind, "cdr"))
        counts: Counter = Counter()
        # mcap merge is a k-way merge on log time: every input must be in order.
        stream = heapq.merge(boxes(), pictures(pool), key=lambda m: m[0])
        for t_ns, name, msg in stream:
            w.write(name, serialize_message(msg), t_ns)
            counts[name] += 1
        del w  # closes the bag; rosbag2_py has no close()
    return counts


# ---------------------------------------------------------------------------
# Merging with the source (mcap CLI).
# ---------------------------------------------------------------------------


def merge_commands(
    src: Path,
    overlay: Path,
    out: Path,
    parts: tuple[Path, Path, Path],
    *,
    window: Optional[tuple[int, int]],
    static_topics: list[str],
    compression: str,
) -> list[list[str]]:
    """The mcap CLI invocations that write out: the source (or, with a window
    [start, end] inclusive on log time, its window part plus its static topics
    whole) merged with the overlay. parts: scratch paths for the window part,
    the static part and the overlay without its metadata."""
    window_part, static_part, overlay_part = parts
    # The overlay's own rosbag2 metadata describes only the new topics and
    # names its scratch file; the output keeps the source's.
    cmds = [
        ["mcap", "filter", str(overlay), "-o", str(overlay_part)]
        + ["--exclude-metadata", "--compression", compression]
    ]
    if window is None:
        inputs = [src]
    else:
        cut = [
            "mcap", "filter", str(src), "-o", str(window_part),
            "--start", str(window[0]),
            "--end", str(window[1] + 1),  # --end is exclusive
            "--compression", compression,
        ]  # fmt: skip
        for t in static_topics:
            cut += ["--exclude-topic-regex", topic_regex(t)]
        cmds.append(cut)
        inputs = [window_part]
        if static_topics:
            static = ["mcap", "filter", str(src), "-o", str(static_part)]
            static += ["--compression", compression]
            for t in static_topics:
                static += ["--include-topic-regex", topic_regex(t)]
            cmds.append(static)
            inputs.append(static_part)
    cmds.append(
        ["mcap", "merge", *map(str, inputs), str(overlay_part), "-o", str(out)]
        + ["--compression", compression]
        # The window and static parts each carry the source's metadata.
        + ["--allow-duplicate-metadata"]
    )
    return cmds


@dataclass
class DebugBagResult:
    output: str
    source: str
    window_ns: Optional[tuple[int, int]]
    frames: int
    frames_without_pose: int
    images: int  # camera images drawn on, per image layer
    boxes: dict[str, int]  # topic -> boxes over all frames
    messages: dict[str, int]  # topic -> messages written


def source_bag(bag_id: str) -> Path:
    """The bag ingest read (bag_meta.json's source_path)."""
    with open(local_path(bag_meta_path(bag_id)), encoding="utf-8") as fh:
        return resolve_mcap_file(json.load(fh)["source_path"])


def default_output(src: Path, out_dir: str, chunk_id: Optional[str]) -> Path:
    """<out_dir>/<source stem>[_<chunk>]_debug.mcap"""
    stem = src.stem
    if (src.parent / "metadata.yaml").exists():
        stem = src.parent.name  # a rosbag2 directory names the bag
    if chunk_id is not None:
        stem += f"_{chunk_id}"
    return Path(out_dir) / f"{stem}_debug.mcap"


def chunk_window(bag_id: str, chunk_id: str) -> tuple[int, int]:
    """The chunk's decoded window [t_overlap_start_ns, t_overlap_end_ns]."""
    for r in read_rows(chunks_index_path(bag_id)):
        if r["chunk_id"] == chunk_id:
            return int(r["t_overlap_start_ns"]), int(r["t_overlap_end_ns"])
    raise DebugBagError(f"bag {bag_id!r} has no chunk {chunk_id!r}")


def debug_bag(
    bag_id: str,
    out_path: str,
    *,
    chunk_id: Optional[str] = None,
    source: Optional[str] = None,
    static_topics: tuple[str, ...] = (),
    compression: str = "none",
    overwrite: bool = False,
    image_layers: tuple[str, ...] = (),
    workers: int = 1,
) -> DebugBagResult:
    """Write the source bag plus the box topics, and the image_layers
    (debug_images.LAYERS) per camera, to out_path."""
    unknown = set(image_layers) - set(debug_images.LAYERS)
    if unknown:
        raise DebugBagError(f"unknown image layers {sorted(unknown)}")
    src = resolve_mcap_file(source) if source else source_bag(bag_id)
    out = Path(out_path)
    if out.exists() and not overwrite:
        raise DebugBagError(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)

    frame_id = Calibration.load(bag_id).ego_frame
    if not frame_id:
        raise DebugBagError(f"bag {bag_id!r}: calibration.json names no ego_frame")
    frames = load_frames(bag_id, chunk_id)
    if not frames:
        raise DebugBagError(f"bag {bag_id!r}: no frames to write")
    images = load_images(bag_id, chunk_id, image_layers) if image_layers else []
    window = chunk_window(bag_id, chunk_id) if chunk_id is not None else None

    overlay_dir = out.with_name(f".{out.stem}.overlay")
    tmp = out.with_name(f".{out.stem}.partial.mcap")
    parts = (
        out.with_name(f".{out.stem}.window.mcap"),
        out.with_name(f".{out.stem}.static.mcap"),
        out.with_name(f".{out.stem}.overlay.mcap"),
    )
    shutil.rmtree(overlay_dir, ignore_errors=True)
    try:
        counts = write_overlay(
            overlay_dir, frames, frame_id, images, tuple(image_layers), workers
        )
        cmds = merge_commands(
            src,
            resolve_mcap_file(str(overlay_dir)),
            tmp,
            parts,
            window=window,
            static_topics=list(static_topics),
            compression=compression,
        )
        for cmd in cmds:
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise DebugBagError(f"{' '.join(cmd)} failed:\n{done.stderr.strip()}")
        tmp.replace(out)
    finally:
        shutil.rmtree(overlay_dir, ignore_errors=True)
        for p in (tmp, *parts):
            p.unlink(missing_ok=True)

    boxes: Counter = Counter()
    for f in frames:
        for layer, bs in f.boxes.items():
            boxes[topic(layer)] += len(bs)
    return DebugBagResult(
        output=str(out),
        source=str(src),
        window_ns=window,
        frames=len(frames),
        frames_without_pose=sum(not f.has_pose for f in frames),
        images=len(images),
        boxes=dict(sorted(boxes.items())),
        messages=dict(sorted(counts.items())),
    )
