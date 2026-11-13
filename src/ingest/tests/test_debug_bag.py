"""debug-bag: proposals, tracks and refined labels as ego-frame MarkerArrays
merged into the source bag."""

import math
import shutil
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wato_common.artifacts.proposals import Proposal
from wato_common.artifacts.refined_labels import RefinedBox
from wato_common.artifacts.tracks import TrackFrame, TrackSummary
from wato_common.geometry.interpolation import PoseSample
from wato_common.pose_lookup import PoseLookup
from wato_ingest import debug_bag

MS = 1_000_000
BOX = dict(cx=10.0, cy=7.0, cz=1.0, w=1.8, l=4.5, h=1.5, heading=math.pi / 2)


def _proposal(pid, prov, t_ms, sweep_id=0, **kw):
    base = dict(
        proposal_id=pid,
        sweep_id=sweep_id,
        frame_id=sweep_id,
        sweep_ids=(sweep_id,),
        reference_timestamp_ns=t_ms * MS,
        cls="car",
        score=0.8,
        provenance=prov,
        lidar_point_count=120,
    )
    return Proposal(**(base | BOX | kw))


def _track_frame(track_id, t_ms, sweep_id, chunk_id="0000"):
    return TrackFrame(
        track_id=track_id,
        chunk_id=chunk_id,
        sweep_id=sweep_id,
        reference_timestamp_ns=t_ms * MS,
        proposal_id=f"fused-{track_id}-{sweep_id}",
        cls="car",
        score=0.8,
        owned=True,
        **BOX,
    )


def _summary(track_id, state="moving"):
    return TrackSummary(
        track_id=track_id,
        n_frames=2,
        first_timestamp_ns=0,
        last_timestamp_ns=1,
        motion_state=state,
        cls="car",
        n_strong=1,
        frac_two_source=0.5,
        frac_lidar_mos=0.5,
        net_displacement_m=3.0,
        centre_var_m2=0.1,
        median_points=100.0,
    )


def _refined(track_id, t_ms, sweep_id, directed=True):
    return RefinedBox(
        track_id=track_id,
        chunk_id="0000",
        sweep_id=sweep_id,
        reference_timestamp_ns=t_ms * MS,
        heading_directed=directed,
        cls="car",
        motion_state="moving",
        size_source="slf",
        confidence=0.7,
        owned=True,
        **BOX,
    )


def _drawn(layer="proposals/fused", t_ms=0, sweep_id=0, **box):
    return debug_bag.Drawn(
        layer=layer,
        sweep_id=sweep_id,
        t_ns=t_ms * MS,
        box=debug_bag.box_of(_proposal("p", "fused", t_ms, **box)),
        text="",
        rgb=debug_bag.CLASS_RGB["car"],
    )


def _ego_at(x, y, yaw):
    """world_T_ego for an ego at (x, y, 0) facing yaw."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = (x, y, 0.0)
    return T


def _poses(*t_ms, x=10.0, y=5.0, yaw=math.pi / 2):
    q = Rotation.from_euler("z", yaw).as_quat()
    return PoseLookup([PoseSample(t * MS, np.array([x, y, 0.0]), q) for t in t_ms])


def test_ego_box_is_the_world_box_seen_from_the_ego():
    # The ego at (10, 5) faces +y; the car 2 m ahead of it points along +y too.
    b = debug_bag.ego_box(_drawn(), np.linalg.inv(_ego_at(10.0, 5.0, math.pi / 2)))
    np.testing.assert_allclose(b.centre, (2.0, 0.0, 1.0), atol=1e-9)
    # Pointing straight ahead in the ego frame: no rotation (q or -q).
    assert abs(abs(b.quat_xyzw[3]) - 1.0) < 1e-9
    assert b.size == (4.5, 1.8, 1.5)
    assert b.rgba[:3] == debug_bag.CLASS_RGB["car"]


def test_ego_box_keeps_the_ego_tilt():
    # The ego pitched 0.1 rad: a level world box is pitched -0.1 rad in it.
    world_T_ego = np.eye(4)
    world_T_ego[:3, :3] = Rotation.from_euler("y", 0.1).as_matrix()
    d = _drawn(cx=0.0, cy=0.0, cz=0.0, heading=0.0)
    b = debug_bag.ego_box(d, np.linalg.inv(world_T_ego))
    pitch = Rotation.from_quat(b.quat_xyzw).as_euler("xyz")[1]
    assert pitch == pytest.approx(-0.1)


def test_unclassed_box_is_grey():
    (d,) = debug_bag.proposal_items(
        [_proposal("lidar_mos-0", "lidar_mos", 0, cls=None)]
    )
    assert d.rgb == debug_bag.UNCLASSED_RGB and d.layer == "proposals/lidar_mos"


def test_labels_name_what_backs_each_box():
    lift = _proposal(
        "lifted-7-000000", "lifted", 0, lift_confidence=0.61, lift_seed_point_count=40
    )
    mos = _proposal("lidar_mos-3-000000", "lidar_mos", 0, cls=None, motion_score=0.9)
    fused = _proposal(
        "fused-0", "fused", 0, member_proposal_ids=(lift.proposal_id, mos.proposal_id)
    )
    by_id = {p.proposal_id: p for p in (lift, mos, fused)}
    assert debug_bag.label(fused, by_id) == "car 0.80\n[lidar_mos+lifted] n=120"
    assert debug_bag.label(lift, by_id) == "car 0.80\nconf=0.61 seeds=40 n=120"
    assert debug_bag.label(mos, by_id) == "? 0.80\nmotion=0.90 n=120"


def test_tracks_are_coloured_per_track_and_labelled_with_their_state():
    rows = [_track_frame("0000-1", 0, 0), _track_frame("0000-2", 0, 0)]
    a, b = debug_bag.track_items(rows, [_summary("0000-1", "parked")])
    assert a.layer == debug_bag.TRACKS
    assert a.text == "0000-1 car\nparked" and b.text.endswith("\n?")
    assert a.rgb != b.rgb
    assert a.rgb == debug_bag.track_items(rows[:1], [])[0].rgb  # stable per id


def test_refined_labels_draw_an_arrow_only_for_a_directed_heading():
    a, b = debug_bag.refined_items(
        [_refined("0000-1", 0, 0), _refined("0000-2", 0, 0, directed=False)]
    )
    assert a.layer == debug_bag.REFINED and a.text == "0000-1 car 0.70\nmoving slf"
    assert a.directed and not b.directed


def test_chunk_frames_keep_the_window_and_clear_frames_without_a_pose():
    items = debug_bag.proposal_items(
        [
            _proposal("fused-a", "fused", 100, sweep_id=1),
            _proposal("lifted-a-000001", "lifted", 100, sweep_id=1),
            _proposal("fused-b", "fused", 200, sweep_id=2),
            _proposal("fused-c", "fused", 300, sweep_id=3),  # past the window
            _proposal("fused-d", "fused", 50, sweep_id=0),  # before the poses
        ]
    ) + debug_bag.track_items([_track_frame("0000-1", 200, 2)], [])
    layers = ["proposals/lifted", "proposals/fused", debug_bag.TRACKS]
    frames = debug_bag.chunk_frames(
        "0000", items, _poses(100, 250), window=(0, 300 * MS), layers=layers
    )
    assert [(f.sweep_id, f.t_ns) for f in frames] == [
        (0, 50 * MS),
        (1, 100 * MS),
        (2, 200 * MS),
    ]
    no_pose, first, second = frames
    assert not no_pose.has_pose and all(v == () for v in no_pose.boxes.values())
    assert [len(first.boxes[k]) for k in layers] == [1, 1, 0]
    # A layer with no box in a frame still gets a (clearing) entry.
    assert [len(second.boxes[k]) for k in layers] == [0, 1, 1]


def test_whole_bag_is_the_source_merged_with_the_stripped_overlay(tmp_path):
    src, ov, out = tmp_path / "in.mcap", tmp_path / "ov.mcap", tmp_path / "out.mcap"
    w, s, o = tmp_path / "w", tmp_path / "s", tmp_path / "o"
    cmds = debug_bag.merge_commands(
        src, ov, out, (w, s, o),
        window=None, static_topics=[], compression="none",
    )  # fmt: skip
    assert cmds == [
        ["mcap", "filter", str(ov), "-o", str(o), "--exclude-metadata",
         "--compression", "none"],
        ["mcap", "merge", str(src), str(o), "-o", str(out), "--compression",
         "none", "--allow-duplicate-metadata"],
    ]  # fmt: skip


def test_chunk_window_cuts_the_source_and_keeps_static_topics(tmp_path):
    src, ov, out = tmp_path / "in.mcap", tmp_path / "ov.mcap", tmp_path / "out.mcap"
    w, s, o = tmp_path / "w", tmp_path / "s", tmp_path / "o"
    _, cut, static, merge = debug_bag.merge_commands(
        src, ov, out, (w, s, o),
        window=(100, 200), static_topics=["/tf_static"], compression="zstd",
    )  # fmt: skip
    assert cut[:5] == ["mcap", "filter", str(src), "-o", str(w)]
    assert cut[cut.index("--start") + 1] == "100"
    assert cut[cut.index("--end") + 1] == "201"  # --end is exclusive
    assert cut[cut.index("--exclude-topic-regex") + 1] == "^/tf_static$"
    assert static[static.index("--include-topic-regex") + 1] == "^/tf_static$"
    assert merge[:6] == ["mcap", "merge", str(w), str(s), str(o), "-o"]


def test_default_output_names_the_bag_and_chunk(tmp_path):
    f = tmp_path / "rec_0.mcap"
    assert debug_bag.default_output(f, "/o", None).name == "rec_0_debug.mcap"
    assert debug_bag.default_output(f, "/o", "0003").name == "rec_0_0003_debug.mcap"
    d = tmp_path / "bagdir"
    d.mkdir()
    (d / "metadata.yaml").touch()
    assert (
        debug_bag.default_output(d / "bagdir_0.mcap", "/o", None).name
        == "bagdir_debug.mcap"
    )


def _camera_row(cam, seq, header_ms, log_ms, valid=True):
    return dict(
        cam_id=cam,
        camera_seq=seq,
        image_path=f"file:///art/cam_{cam}/{seq:06d}.jpg",
        header_timestamp_ns=header_ms * MS,
        record_timestamp_ns=log_ms * MS,
        valid=valid,
    )


class _Masklets:
    def __init__(self, at):
        self._at = at

    def at(self, cam, seq):
        return self._at.get((cam, seq), [])


def test_chunk_images_are_the_owned_images_with_their_source_stamps():
    rows = [
        _camera_row("CAM_A", 0, 90, 120),  # before the window
        _camera_row("CAM_A", 1, 140, 166),
        _camera_row("CAM_B", 1, 150, 171),
        _camera_row("CAM_B", 2, 190, 215, valid=False),
        _camera_row("CAM_A", 2, 200, 226),  # the window's end is the next chunk's
    ]
    index = _Masklets({("CAM_A", 1): ["m-1", "m-2"]})
    frame_ids = {"CAM_A": "cam_a", "CAM_B": "cam_b"}
    images = debug_bag.chunk_images(
        "b", "0000", rows, (100 * MS, 200 * MS), frame_ids, index
    )
    assert [(f.cam_id, f.camera_seq) for f in images] == [("CAM_A", 1), ("CAM_B", 1)]
    a, b = images
    assert (a.header_ns, a.log_ns, a.frame_id) == (140 * MS, 166 * MS, "cam_a")
    assert a.image_path == "/art/cam_CAM_A/000001.jpg"
    assert a.masklets == ("m-1", "m-2") and b.masklets == ()
    with pytest.raises(debug_bag.DebugBagError, match="CAM_C"):
        debug_bag.chunk_images(
            "b", "0000", [_camera_row("CAM_C", 0, 150, 160)], (0, 1e18), frame_ids
        )


def test_image_topics_name_the_layer_and_camera():
    assert debug_bag.image_topic("masks", "PANO_NN") == "/wato_world/masks/pano_nn"


# ---------------------------------------------------------------------------
# End to end: needs rosbag2 (ingest container) and the mcap CLI.
# ---------------------------------------------------------------------------
def _source_bag(path):
    rosbag2_py = pytest.importorskip("rosbag2_py")
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Imu

    w = rosbag2_py.SequentialWriter()
    w.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    w.create_topic(rosbag2_py.TopicMetadata(0, "/imu", "sensor_msgs/msg/Imu", "cdr"))
    for t in range(0, 401, 50):
        w.write("/imu", serialize_message(Imu()), t * MS)
    del w


def _read(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    from visualization_msgs.msg import MarkerArray

    r = rosbag2_py.SequentialReader()
    r.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    msgs = []
    while r.has_next():
        topic, data, t = r.read_next()
        if types[topic] == debug_bag.MARKER_ARRAY:
            data = deserialize_message(data, MarkerArray)
        elif types[topic] == debug_bag.COMPRESSED_IMAGE:
            data = deserialize_message(data, CompressedImage)
        msgs.append((topic, t, data))
    return types, msgs


@pytest.mark.skipif(shutil.which("mcap") is None, reason="mcap CLI not installed")
def test_overlay_merges_into_the_source(tmp_path):
    _source_bag(tmp_path / "src")
    items = (
        debug_bag.proposal_items(
            [
                _proposal("fused-a", "fused", 100, sweep_id=1),
                _proposal("lifted-a-000001", "lifted", 100, sweep_id=1),
                _proposal("fused-b", "fused", 200, sweep_id=2),
            ]
        )
        + debug_bag.track_items([_track_frame("0000-1", 100, 1)], [_summary("0000-1")])
        + debug_bag.refined_items([_refined("0000-1", 100, 1)])
    )
    layers = ["proposals/lifted", "proposals/fused", "tracks", "refined_labels"]
    frames = debug_bag.chunk_frames(
        "0000", items, _poses(0, 400), window=(-math.inf, math.inf), layers=layers
    )
    counts = debug_bag.write_overlay(tmp_path / "overlay", frames, "base_link")
    assert counts == {f"/wato_world/{layer}": 2 for layer in layers}
    out = tmp_path / "out.mcap"
    for cmd in debug_bag.merge_commands(
        debug_bag.resolve_mcap_file(str(tmp_path / "src")),
        debug_bag.resolve_mcap_file(str(tmp_path / "overlay")),
        out,
        (tmp_path / "w.mcap", tmp_path / "s.mcap", tmp_path / "o.mcap"),
        window=None,
        static_topics=[],
        compression="none",
    ):
        subprocess.run(cmd, check=True, capture_output=True)

    meta = subprocess.run(
        ["mcap", "list", "metadata", str(out)], check=True, capture_output=True
    ).stdout.decode()
    assert "overlay" not in meta  # only the source's rosbag2 record
    types, msgs = _read(out)
    assert types["/imu"] == "sensor_msgs/msg/Imu"
    assert types["/wato_world/refined_labels"] == debug_bag.MARKER_ARRAY
    assert sum(m[0] == "/imu" for m in msgs) == 9

    def at(topic):
        return [(t, m) for name, t, m in msgs if name == topic]

    fused = at("/wato_world/proposals/fused")
    assert [t for t, _ in fused] == [100 * MS, 200 * MS]
    markers = fused[0][1].markers
    # DELETEALL per namespace, then the box's cube and label.
    assert [(m.ns, m.action) for m in markers[:3]] == [
        ("boxes", 3),
        ("labels", 3),
        ("headings", 3),
    ]
    cube, text = markers[3], markers[4]
    assert cube.header.frame_id == "base_link"
    assert (cube.header.stamp.sec, cube.header.stamp.nanosec) == (0, 100 * MS)
    assert (cube.pose.position.x, cube.pose.position.y) == pytest.approx((2.0, 0.0))
    assert text.text.startswith("car 0.80")
    # The refined box's directed heading gets an arrow; the track's box doesn't.
    refined = at("/wato_world/refined_labels")[0][1].markers
    assert [m.ns for m in refined[3:]] == ["boxes", "labels", "headings"]
    assert [m.ns for m in at("/wato_world/tracks")[0][1].markers[3:]] == [
        "boxes",
        "labels",
    ]
    # Frames where a layer has no box only clear it.
    assert len(at("/wato_world/tracks")[1][1].markers) == 3


@pytest.mark.skipif(shutil.which("mcap") is None, reason="mcap CLI not installed")
def test_images_replay_with_their_source_images(tmp_path, monkeypatch):
    """Each camera image's drawn layers carry its header stamp, log time and
    frame_id, interleaved with the boxes in log-time order."""
    from PIL import Image

    from wato_common.artifacts import masklets
    from wato_common.artifacts.masklets import MaskletRecord
    from wato_ingest import debug_images

    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")
    _source_bag(tmp_path / "src")
    m = np.zeros((48, 64), bool)
    m[10:30, 10:40] = True
    masklets.write_mask("b", "0000", "m-1", 0, m)
    rec = MaskletRecord(
        masklet_id="m-1", cam_id="CAM_A", cls="car", score=0.9, frames_present=(0,)
    )
    images = []
    for cam, seq, header_ms, log_ms in [
        ("CAM_A", 0, 95, 120),
        ("CAM_B", 0, 98, 121),
        ("CAM_A", 1, 145, 170),
    ]:
        path = tmp_path / f"{cam}_{seq}.jpg"
        Image.fromarray(np.full((48, 64, 3), 100, np.uint8)).save(path)
        images.append(
            debug_images.ImageFrame(
                bag_id="b",
                chunk_id="0000",
                cam_id=cam,
                camera_seq=seq,
                image_path=str(path),
                header_ns=header_ms * MS + 7,
                log_ns=log_ms * MS,
                frame_id=cam.lower(),
                masklets=(rec,) if (cam, seq) == ("CAM_A", 0) else (),
            )
        )
    frames = debug_bag.chunk_frames(
        "0000",
        debug_bag.proposal_items(
            [
                _proposal("fused-a", "fused", 100, sweep_id=1),
                _proposal("fused-b", "fused", 150, sweep_id=2),
            ]
        ),
        _poses(0, 400),
        window=(-math.inf, math.inf),
        layers=["proposals/fused"],
    )
    counts = debug_bag.write_overlay(
        tmp_path / "overlay",
        frames,
        "base_link",
        images,
        (debug_images.MASKS, debug_images.DEPTH),
        workers=2,
    )
    assert counts == {
        "/wato_world/proposals/fused": 2,
        "/wato_world/masks/cam_a": 2,
        "/wato_world/masks/cam_b": 1,
        "/wato_world/depth/cam_a": 2,
        "/wato_world/depth/cam_b": 1,
    }
    out = tmp_path / "out.mcap"
    for cmd in debug_bag.merge_commands(
        debug_bag.resolve_mcap_file(str(tmp_path / "src")),
        debug_bag.resolve_mcap_file(str(tmp_path / "overlay")),
        out,
        (tmp_path / "w.mcap", tmp_path / "s.mcap", tmp_path / "o.mcap"),
        window=None,
        static_topics=[],
        compression="none",
    ):
        subprocess.run(cmd, check=True, capture_output=True)
    types, msgs = _read(out)
    assert types["/wato_world/masks/cam_a"] == debug_bag.COMPRESSED_IMAGE
    mine = [(name, t) for name, t, _ in msgs if name.startswith("/wato_world/")]
    assert [t for _, t in mine] == sorted(t for _, t in mine)
    masks = [(t, msg) for name, t, msg in msgs if name == "/wato_world/masks/cam_a"]
    (t0, first), (t1, _) = masks
    assert (t0, t1) == (120 * MS, 170 * MS)  # the source images' log times
    stamp = first.header.stamp.sec * 1_000_000_000 + first.header.stamp.nanosec
    assert stamp == 95 * MS + 7 and first.header.frame_id == "cam_a"
    assert first.format == debug_bag.JPEG_FORMAT
    assert bytes(first.data[:2]) == b"\xff\xd8"  # a JPEG
