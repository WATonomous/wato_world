"""A two-frame, two-LiDAR chunk with every upstream artifact
proposal_generation reads, written through the producers' wato_common writers.

Scene, world == ego, ground flat at z = 0, in both frames:
  A  a moving car at x = 10: a gated Step F cluster AND lifted instance "mA"
     (car) -> one fused pair; one "mA" point bleeds onto background at (30, 30)
  B  a parked car at y = 10: lifted instance "mB" (car) only
  C  a mover at x = -10: a gated cluster with no lifted labels -> class-less
  D  a static blob at y = -10: a cluster failing the motion gate
Frame k: lidar_cc sweep 2k at 100(k+1) ms (canonical), lidar_ne sweep 2k+1
5 ms later. Car points sit 0.3 m and up, above Step F's height floor.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from wato_common.artifact_store import (
    camera_image_path,
    chunks_index_path,
    lidar_proc_dir,
    lidar_proc_index_path,
    local_path,
    poses_path,
)
from wato_common.artifacts import (
    calibration,
    frame_index,
    ground,
    lidar_sweeps,
    lifted_candidates,
    lifted_labels,
    motion_proposals,
)
from wato_common.artifacts.calibration import (
    Calibration,
    CameraCalibration,
    LidarCalibration,
)
from wato_common.io.parquet_io import write_table
from wato_common.schemas import (
    CHUNK_SCHEMA,
    POSES_SCHEMA,
    PROCESSED_SWEEPS_SCHEMA,
    FrameIndexRow,
    LiftedStatsRow,
    ProcessedSweepMeta,
)
from wato_common.tracking import fit_bev_box

MS = 1_000_000
BAG, CHUNK, CAM = "b", "0000", "CAM_FRONT"
N_FRAMES = 2
W, H = 64, 48
K = [[20.0, 0.0, 32.0], [0.0, 20.0, 24.0], [0.0, 0.0, 1.0]]
# Camera looking along ego +x: optical z = ego x, x = -ego y, y = -ego z.
EGO_T_CAM = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
CENTERS = {"A": (10.0, 0.0), "B": (0.0, 10.0), "C": (-10.0, 0.0), "D": (0.0, -10.0)}
BLEED = np.array([[30.0, 30.0, 1.0]])


def sweep_id(frame: int, lidar: str) -> int:
    return 2 * frame + (lidar == "lidar_ne")


def t_ns(frame: int, lidar: str) -> int:
    return (100 * (frame + 1) + 5 * (lidar == "lidar_ne")) * MS


def _blob(center: tuple[float, float], seed: int, n: int = 40) -> np.ndarray:
    """A car-sized block of points, 4 x 2 m, z 0.3-1.5 m."""
    rng = np.random.default_rng(seed)
    local = rng.uniform([-2.0, -1.0, 0.3], [2.0, 1.0, 1.5], size=(n, 3))
    return local + np.array([center[0], center[1], 0.0])


def stage_chunk(*, with_camera: bool = False) -> None:
    write_table(
        [
            {
                "bag_id": BAG,
                "chunk_id": CHUNK,
                "t_start_ns": 0,
                "t_end_ns": 400 * MS,
                "t_overlap_start_ns": 0,
                "t_overlap_end_ns": 400 * MS,
            }
        ],
        CHUNK_SCHEMA,
        chunks_index_path(BAG),
    )
    os.makedirs(local_path(lidar_proc_dir(BAG, CHUNK)), exist_ok=True)
    normals = np.zeros((40, 40, 3), np.float32)
    normals[..., 2] = 1.0
    ground.write(
        BAG,
        CHUNK,
        ground.GroundGrid(
            np.zeros((40, 40), np.float32), normals, np.array([-50.0, -50.0]), 2.5
        ),
        np.empty((0, 3)),
    )

    meta, clusters, stats = [], [], []
    for f in range(N_FRAMES):
        for lidar in ("lidar_cc", "lidar_ne"):
            sid = sweep_id(f, lidar)
            seed = 10 * sid
            parts = {
                name: _blob(c, seed + k) for k, (name, c) in enumerate(CENTERS.items())
            }
            xyz = np.concatenate([parts[n] for n in "ABCD"] + [BLEED])
            offsets = np.cumsum([0] + [len(parts[n]) for n in "ABCD"])
            rows_of = {
                n: np.arange(offsets[k], offsets[k + 1]) for k, n in enumerate("ABCD")
            }
            bleed_row = len(xyz) - 1

            world_uri = lidar_sweeps.write_world(
                BAG,
                CHUNK,
                sid,
                lidar_sweeps.WorldSweep(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    origin=np.zeros(3),
                    ground_mask=np.zeros(len(xyz), bool),
                ),
            )
            mask_uri = lidar_sweeps.write_dynamic_mask(
                BAG, CHUNK, sid, np.zeros(len(xyz), bool)
            )
            meta.append(
                ProcessedSweepMeta(
                    bag_id=BAG,
                    chunk_id=CHUNK,
                    sweep_id=sid,
                    lidar_id=lidar,
                    frame_id=f,
                    reference_timestamp_ns=t_ns(f, lidar),
                    n_points_total=len(xyz),
                    n_points_static=len(xyz),
                    n_points_dynamic=0,
                    world_path=world_uri,
                    dynamic_mask_path=mask_uri,
                    has_intensity=False,
                    deskewed=True,
                ).model_dump()
            )

            cluster_id = np.full(len(xyz), -1, np.int32)
            for k, name in enumerate("ACD"):
                cluster_id[rows_of[name]] = 3 * f + k
            motion_proposals.write_sweep(
                BAG,
                CHUNK,
                sid,
                np.where(cluster_id >= 0, motion_proposals.AW_DYNAMIC, 0).astype(
                    np.uint8
                ),
                cluster_id,
            )

            labeled = np.concatenate([rows_of["A"], [bleed_row], rows_of["B"]])
            ids = ["mA"] * (len(rows_of["A"]) + 1) + ["mB"] * len(rows_of["B"])
            lifted_labels.write_sweep(
                BAG,
                CHUNK,
                sid,
                lifted_labels.SweepLabels(
                    point_indices=labeled.astype(np.int32),
                    instance_ids=np.array(ids),
                    classes=np.array(["car"] * len(ids)),
                    confidences=np.full(len(ids), 0.9, np.float32),
                    n_supporting_cameras=np.ones(len(ids), np.int32),
                    n_disagreeing_cameras=np.zeros(len(ids), np.int32),
                ),
            )
            lifted_candidates.write_sweep(
                BAG,
                CHUNK,
                sid,
                lifted_candidates.SweepCandidates(
                    labeled.astype(np.int32),
                    np.array(ids),
                    np.full(len(ids), 0.9, np.float32),
                ),
            )
            stats.append(
                LiftedStatsRow(
                    sweep_id=sid,
                    bag_id=BAG,
                    chunk_id=CHUNK,
                    n_points_total=len(xyz),
                    n_points_labeled=len(ids),
                    n_points_in_any_mask=len(ids),
                    n_points_failed_visibility=0,
                    n_points_disagreement=0,
                    mean_confidence_labeled=0.9,
                    n_cameras_used=1,
                )
            )

        # Step F's clusters: one per object per frame, its seed box over the
        # frame's points, stamped with the canonical sweep's time.
        for k, (name, score) in enumerate((("A", 2.0), ("C", 1.6), ("D", 0.2))):
            box = fit_bev_box(
                np.concatenate(
                    [
                        _blob(CENTERS[name], 10 * sweep_id(f, lid) + "ABCD".index(name))
                        for lid in ("lidar_cc", "lidar_ne")
                    ]
                )
            )
            clusters.append(
                motion_proposals.MotionCluster(
                    frame_id=f,
                    reference_timestamp_ns=t_ns(f, "lidar_cc"),
                    sweep_ids=(sweep_id(f, "lidar_cc"), sweep_id(f, "lidar_ne")),
                    cluster_id=3 * f + k,
                    track_hint_id=k,
                    cx=box.cx,
                    cy=box.cy,
                    cz=box.cz,
                    w=box.w,
                    l=box.l,
                    h=box.h,
                    heading=box.heading,
                    n_points=80,
                    n_seed_points=80,
                    source_bits=int(
                        motion_proposals.AW_DYNAMIC | motion_proposals.SEG_DYNAMIC
                    ),
                    n_sources=2,
                    frac_seg_dynamic=0.8,
                    frac_persistent=0.05,
                    track_life=N_FRAMES,
                    net_displacement_m=score * 4.0,
                    motion_score=score,
                    box_filled=score > 1,
                )
            )

    write_table(meta, PROCESSED_SWEEPS_SCHEMA, lidar_proc_index_path(BAG, CHUNK))
    motion_proposals.write_clusters(BAG, CHUNK, clusters)
    lifted_labels.write_instances(
        BAG,
        CHUNK,
        [
            lifted_labels.LiftedInstance(
                instance_id=iid,
                cls="car",
                masklet_ids=(iid,),
                cam_ids=(CAM,),
                n_points_labeled=200,
                n_sweeps=4,
            )
            for iid in ("mA", "mB")
        ],
    )
    lifted_labels.write_stats(BAG, CHUNK, stats)
    if with_camera:
        _stage_camera()


def _stage_camera() -> None:
    """CAM_FRONT, looking along +x at car A, one image per frame."""
    pose = {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    write_table(
        [
            {
                "bag_id": BAG,
                "chunk_id": CHUNK,
                "timestamp_ns": t * MS,
                **pose,
                "world_T_ego_flat": np.eye(4).flatten().tolist(),
                "source": "/odom",
                "valid": True,
                "interval_drop_reason": None,
            }
            for t in (0, 400)
        ],
        POSES_SCHEMA,
        poses_path(BAG, CHUNK),
    )
    rows = []
    for f in range(N_FRAMES):
        image = local_path(camera_image_path(BAG, CHUNK, CAM, f, "jpg"))
        os.makedirs(os.path.dirname(image), exist_ok=True)
        Image.new("RGB", (W, H), (40, 40, 40)).save(image)
        for lidar in ("lidar_cc", "lidar_ne"):
            rows.append(
                FrameIndexRow(
                    frame_id=f"f{f}{lidar}",
                    bag_id=BAG,
                    chunk_id=CHUNK,
                    sweep_id=sweep_id(f, lidar),
                    lidar_id=lidar,
                    lidar_path="",
                    reference_timestamp_ns=t_ns(f, lidar),
                    cam_id=CAM,
                    image_path=image,
                    camera_seq=f,
                    camera_timestamp_ns=t_ns(f, "lidar_cc"),
                    valid_camera=True,
                    valid_pose=True,
                )
            )
    frame_index.write(BAG, CHUNK, rows)
    calibration.write(
        BAG,
        Calibration(
            cameras={
                CAM: CameraCalibration(
                    cam_id=CAM, K=np.array(K), ego_T_cam=np.array(EGO_T_CAM, float)
                )
            },
            lidars={
                lid: LidarCalibration(lid, np.eye(4))
                for lid in ("lidar_cc", "lidar_ne")
            },
        ),
    )
