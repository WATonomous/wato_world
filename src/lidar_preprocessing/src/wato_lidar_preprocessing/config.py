"""Pydantic-loaded config for lidar_preprocessing.

Every physical constant — log-odds increments, decision thresholds, carve
geometry, scan rate, FoV, beam count, intensity scale — is derived from the
datasheet SensorModel selected by ``sensor_model`` (see sensor_model.py).
YAML picks scanners and states the few genuinely free choices (voxel size,
evidence count, which stages run). It does not state physics.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from wato_lidar_preprocessing.sensor_model import SensorModel, get_sensor_model


class SensorModelParams(BaseModel):
    """Selects a datasheet scanner profile, optionally one per lidar_id.

    The physical numbers live in sensor_model.py's profile table, never in
    user YAML. ``profile`` is the rig default; ``per_lidar`` overrides it for
    a mixed rig (the WATO car runs a VLP-32C centre and two VLP-16 corners),
    so each scanner's own FoV, beam count, range and rate are used wherever
    the work is per-sweep.

    The default profile also supplies the chunk-level decision constants
    (l_occ, l_free, thresholds), which are shared across profiles by
    construction — see sensor_model.py — so a mixed rig does not classify
    one lidar's returns by a different rule than another's.
    """

    model_config = ConfigDict(extra="forbid")

    profile: str = "vlp32c"
    per_lidar: dict[str, str] = Field(default_factory=dict)

    @field_validator("profile")
    @classmethod
    def _known_profile(cls, v: str) -> str:
        get_sensor_model(v)  # raises ValueError listing valid profiles
        return v

    @field_validator("per_lidar")
    @classmethod
    def _known_per_lidar(cls, v: dict[str, str]) -> dict[str, str]:
        for lidar_id, prof in v.items():
            try:
                get_sensor_model(prof)
            except ValueError as exc:
                raise ValueError(f"per_lidar[{lidar_id!r}]: {exc}") from None
        return v

    def build(self, lidar_id: Optional[str] = None) -> SensorModel:
        """The profile for ``lidar_id``, falling back to the rig default."""
        if lidar_id is not None and lidar_id in self.per_lidar:
            return get_sensor_model(self.per_lidar[lidar_id])
        return get_sensor_model(self.profile)


class FrameSyncParams(BaseModel):
    """Multi-lidar sweep-to-frame grouping for SAM4D-style downstream fusion.

    A "frame" is one tick of canonical_lidar. Non-canonical sweeps within
    ±tolerance_ms inherit the canonical frame_id.

    canonical_lidar=None disables grouping — each sweep becomes its own
    frame indexed sequentially per lidar_id (correct for single-lidar bags).
    """

    model_config = ConfigDict(extra="forbid")

    canonical_lidar: Optional[str] = None
    tolerance_ms: float = 25.0

    @field_validator("tolerance_ms")
    @classmethod
    def _positive_tolerance(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"tolerance_ms must be > 0, got {v}")
        return v

    def tolerance_ns(self) -> int:
        return int(self.tolerance_ms * 1_000_000)


class PatchworkParams(BaseModel):
    """Patchwork++ ground-segmentation parameters.

    Defaults mirror wato_monorepo perception/patchwork/patchwork/config/params.yaml.
    """

    model_config = ConfigDict(extra="forbid")

    sensor_height: float = 1.8
    num_iter: int = 3
    num_lpr: int = 20
    num_min_pts: int = 10
    th_seeds: float = 0.3
    th_dist: float = 0.15
    th_seeds_v: float = 0.25
    th_dist_v: float = 0.85
    max_range: float = 90.0
    min_range: float = 1.0
    uprightness_thr: float = 0.101
    enable_RNR: bool = False
    verbose: bool = False
    ground_cell_size_m: float = 0.25

    def to_patchwork_dict(self) -> dict[str, Any]:
        """Return kwargs accepted by pypatchworkpp.patchworkpp()."""
        return {
            "sensor_height": self.sensor_height,
            "num_iter": self.num_iter,
            "num_lpr": self.num_lpr,
            "num_min_pts": self.num_min_pts,
            "th_seeds": self.th_seeds,
            "th_dist": self.th_dist,
            "th_seeds_v": self.th_seeds_v,
            "th_dist_v": self.th_dist_v,
            "max_range": self.max_range,
            "min_range": self.min_range,
            "uprightness_thr": self.uprightness_thr,
            "enable_RNR": self.enable_RNR,
            "verbose": self.verbose,
        }


class MFMosParams(BaseModel):
    """MF-MOS moving-object segmentation parameters.

    Used by the `mos` and `union` segmentation methods (`--seg mos|union`);
    never by `aw`. On `mos` it is fully self-contained: inference (Step A.5)
    plus static/dynamic derived purely from the per-sweep moving masks, with
    no Amanatides-Woo / log-odds involvement. Requires a CUDA GPU for
    realistic data; device="cpu" is for tiny smoke tests only.

    The spherical-projection geometry (range-image height, vertical FoV,
    intensity scale, residual spacing) is NOT configured here — it is read
    from each lidar's sensor profile, and the checkpoint-side constants
    (image width, training range window, KITTI reference rate) live in
    mf_mos/_core.py. What remains below is genuinely about running the model:
    where its weights are, how confident a pixel must be, and how the output
    is cleaned.
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = "/data/models/mf_mos/mf_mos_semantic_kitti.pt"
    arch_config: str = "/data/models/mf_mos/arch_cfg.yaml"
    data_config: str = "/data/models/mf_mos/data_cfg.yaml"
    device: str = "cuda"
    score_threshold: float = 0.5
    save_scores: bool = False
    # Max pose-interpolation gap (ms) used when warping a historical sweep
    # into the current frame for a residual. Gates POSE quality only.
    max_pose_gap_ms: float = 200.0
    # Max sweep-to-sweep time baseline (ms) accepted for a residual channel.
    # Distinct from max_pose_gap_ms: a residual at step k spans k * sweep_dt,
    # which for the longer steps legitimately exceeds the pose-gap cap.
    # Conflating the two zeroed every residual channel whose baseline
    # exceeded max_pose_gap_ms, silently gutting multi-frame MOS. The steps
    # are rate-scaled to KITTI's 10 Hz (mf_mos/_core.residual_steps_for), so
    # the longest baseline is ~n_input_scans * 100 ms (800 ms for the
    # released checkpoint) on any rig; keep this comfortably above it.
    max_residual_gap_ms: float = 1000.0
    # Occlusion gate for unprojecting the per-pixel moving mask back to points.
    occlusion_range_tol_m: float = 1.0
    # Seed each lidar's residual sliding window from the temporally-preceding
    # chunk's sweeps so the first sweeps of a chunk get full residual channels
    # instead of cold-start zeros.
    prime_window_from_prior_chunk: bool = True

    # --- Per-sweep spatial denoise (replaces the chunk-wide vote tier) ---
    # MF-MOS speckle is removed spatially per sweep: cluster moving points on a
    # 26-connected 3D grid and drop clusters below the size floor. Temporal
    # confirmation of a mover is the downstream tracker's job.
    # Cluster grid resolution [m] (~2x voxel keeps an object connected).
    moving_cluster_voxel_m: float = 0.5
    # Min points per moving cluster (a pedestrian at MF-MOS range is well above
    # this; single-sweep mispredictions are 1-few points).
    moving_min_cluster_pts: int = 8

    @field_validator("score_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score_threshold must be in [0, 1], got {v}")
        return v

    @field_validator(
        "occlusion_range_tol_m", "moving_cluster_voxel_m", "max_residual_gap_ms"
    )
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @field_validator("moving_min_cluster_pts")
    @classmethod
    def _positive_min_cluster(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"moving_min_cluster_pts must be >= 1, got {v}")
        return v


class MotionFilterParams(BaseModel):
    """Post-fusion temporal motion filter for the union dynamic cloud.

    Two composable gates applied AFTER union's AW-static / ground vetoes, to
    the accumulated per-chunk dynamic cloud. Both exploit the offline batch
    setting and target *currently-moving* semantics:

      persistence — drop a point whose voxel (persistence_voxel_m) is occupied
                    across >= persistence_max_sweeps distinct sweeps. A moving
                    object crosses a 0.5 m voxel in 1-2 sweeps; static
                    structure MF-MOS mislabelled dwells in the same voxel.
      coherence   — drop a point that doesn't belong to a cluster linking into
                    a track of >= coherence_min_life sweeps. Removes the
                    temporally-incoherent specks that survive persistence.

    Tuning is a recall/precision trade and was set by measurement
    (scripts/compare_seg_dynamic, on the WATO ring-road bag, mover-recall
    proxied by distance-from-static-map):

      persist<5  + coherence : on-static 3.9%,  mover-recall 34%  (over-cuts)
      persist<20 (coherence off): on-static 12.6%, mover-recall 75%  (default)
      no filter               : on-static 58.3%, recall 100%

    The default is recall-biased: persistence alone at a loose threshold. Past
    ~24 sweeps the on-static leakage climbs faster than recall, so 20 is the
    recall-biased operating point before that. The persistence ceiling is
    ~75-80% recall because an extended/slow mover (a 4.5 m car at 5 m/s dwells
    ~9 sweeps in a 0.5 m voxel) is indistinguishable from structure by per-voxel
    occupancy alone. Higher recall needs the learned/tracking signal downstream,
    not more geometry here.

    COHERENCE IS OFF BY DEFAULT. It cuts ~20% of real movers for <1% precision
    on sparse (32-/64-beam) LiDAR — distant/fragmented movers don't form clean
    per-sweep clusters that link into tracks. It is kept as an opt-in denoiser
    for dense clouds. A velocity gate and an MF-MOS-AND-AW-dynamic intersection
    both tested worse still (per-sweep visibility drifts a connected-component
    centroid, faking velocity on static structure), so neither is implemented.

    persistence_max_sweeps and coherence_min_life are sweep COUNTS, not times,
    so scale them with sensor rate (NuScenes 20 Hz vs Velodyne 10 Hz).
    """

    model_config = ConfigDict(extra="forbid")

    # Master switch. False = union writes the raw post-veto dynamic cloud
    # (the pre-filter behaviour), for A/B'ing the filter's contribution.
    enabled: bool = True

    # --- persistence gate (the workhorse) ---
    # Drop a dynamic point if its voxel is occupied across >= this many
    # distinct sweeps. Lower = cleaner but cuts more slow/large movers; higher
    # = more recall but more structure leaks. 20 is the recall-biased default
    # (leakage climbs faster past ~24) on the ring-road bag (sweep COUNT —
    # scale with sensor Hz). 0 disables.
    persistence_max_sweeps: int = 20
    # Voxel edge (m) for the persistence count. Coarser than voxel_size_m so a
    # mover's returns across a sweep still land in one voxel (counts as 1-2
    # sweeps) rather than smearing into a per-voxel count of 1 everywhere.
    persistence_voxel_m: float = 0.5

    # --- coherence gate (opt-in; OFF by default — see class docstring) ---
    # Drop a dynamic point unless its per-sweep cluster links into a track
    # spanning >= this many sweeps. 0 disables (default). Membership only — NOT
    # a velocity test. Hurts recall on sparse LiDAR; enable only on dense clouds.
    coherence_min_life: int = 0
    # Connected-components cell (m) for per-sweep clustering.
    coherence_cell_m: float = 0.4
    # Max centroid step (m) between consecutive sweeps when linking clusters
    # into a track. Generous enough for fast movers at the sensor frame rate.
    coherence_link_gate_m: float = 3.0
    # Per-sweep cluster extent cap (m). Clusters larger than this are treated
    # as static structure (walls/facades), not objects, and never seed a
    # track — the size signal that the failed velocity gate lacked.
    coherence_max_object_m: float = 7.0

    @field_validator(
        "persistence_max_sweeps",
        "coherence_min_life",
    )
    @classmethod
    def _nonneg_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"value must be >= 0, got {v}")
        return v

    @field_validator(
        "persistence_voxel_m",
        "coherence_cell_m",
        "coherence_link_gate_m",
        "coherence_max_object_m",
    )
    @classmethod
    def _positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v


class UnionParams(BaseModel):
    """Fusion parameters for the `union` segmentation method (`--seg union`).

    `union` runs BOTH existing methods and combines them instead of picking
    one (see wato_lidar_preprocessing.union):
      * static cloud  = Amanatides-Woo's static map (classify/) — kept
        verbatim. High precision: a voxel is static only with a preponderance
        of occupied evidence.
      * dynamic cloud = MF-MOS moving points (mf_mos/), with Patchwork++
        ground removed and — by default — every point whose voxel AW
        confirmed static removed. The AW static map is the "comparison" that
        rejects MF-MOS false positives hugging static structure.
    """

    model_config = ConfigDict(extra="forbid")

    # Drop any MF-MOS-moving point whose voxel is in AW's static set. This is
    # the core of the method: AW's static map vetoes MF-MOS false positives
    # that hug static structure. False makes `union` == raw MF-MOS dynamic
    # (still ground-removed), useful for A/B'ing the veto's contribution.
    aw_static_veto: bool = True
    # Recall mode. When True the dynamic cloud is the UNION of MF-MOS-moving
    # and AW's own dynamic verdict (both still ground-removed and, when
    # aw_static_veto, static-vetoed). Off by default: AW dynamics are
    # high-recall/low-precision (they hug static surfaces), so the precision
    # default is MF-MOS-only.
    keep_aw_dynamic: bool = False
    # Veto exemption for high-confidence MF-MOS movers. AW's static evidence
    # aggregates the whole chunk, so an object parked for most of the chunk
    # that drives off gets voxel-static evidence — and the veto would delete
    # MF-MOS's correct moving verdict. Points whose MF-MOS moving probability
    # is >= this value survive the veto. Requires mf_mos.save_scores (the
    # exemption is a per-sweep no-op when the score artifact is missing).
    # None disables the exemption (every static-voxel point is vetoed).
    veto_score_exempt: float | None = None
    # Drop dynamic candidates below this height (m) over Step C's ground.npz
    # height grid. The AW static set is structurally blind to the road
    # (skip_endpoint keeps road voxels at n_hits==0, never static), so MF-MOS
    # road false positives — including below-grade artifacts — pass every
    # voxel veto; height over the aggregated ground surface catches them.
    # Costs the bottom slice of real movers (wheels/feet below the
    # threshold). 0.0 disables. Skipped with a warning when ground.npz is
    # missing or a sentinel.
    ground_height_veto_m: float = 0.25
    # Dilate the AW-static veto to voxels within this Chebyshev distance of
    # a static voxel. Surface returns straddle voxel boundaries, so the
    # exact-voxel test misses the leakage shell one voxel off the structure.
    # Candidates whose own voxel AW classed dynamic (dynamic_voxel_keys) are
    # exempt from the dilated part — AW corroborates the motion there.
    # 0 = exact-voxel veto only.
    veto_dilation_voxels: int = 1
    # Post-veto temporal motion filter (persistence + coherence gates). This
    # is what removes the static-structure leakage the voxel vetoes miss —
    # MF-MOS false positives on structure the AW static map covers only
    # sparsely (far walls, foliage). See MotionFilterParams.
    motion_filter: MotionFilterParams = MotionFilterParams()

    @field_validator("veto_score_exempt")
    @classmethod
    def _exempt_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"veto_score_exempt must be in [0, 1], got {v}")
        return v

    @field_validator("ground_height_veto_m")
    @classmethod
    def _ground_height_nonneg(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"ground_height_veto_m must be >= 0, got {v}")
        return v

    @field_validator("veto_dilation_voxels")
    @classmethod
    def _dilation_range(cls, v: int) -> int:
        # (2D+1)^3 - 1 neighbour lookups per candidate; D=3 is already 342.
        if not (0 <= v <= 3):
            raise ValueError(f"veto_dilation_voxels must be in [0, 3], got {v}")
        return v


class ComponentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # --- Sensor model: source of all derived classifier constants ----------
    sensor_model: SensorModelParams = SensorModelParams()

    # Segmentation method for the static/dynamic split (Step B). The two base
    # methods are fully independent modules with no cross-imports:
    #   "aw"    — Amanatides-Woo log-odds voxel ray-casting (classify/). Never
    #             touches MF-MOS; no model inference.
    #   "mos"   — MF-MOS learned moving-object segmentation (mf_mos/). Runs the
    #             model and derives static/dynamic purely from the per-sweep
    #             masks; no ray traversal.
    #   "union" — fusion (union/): runs BOTH, keeps AW's static map, and takes
    #             the dynamic cloud from MF-MOS vetoed by that static map. The
    #             only method allowed to import from the other two. See
    #             UnionParams.
    # Selected at the CLI with `--seg aw|mos|union`.
    segmentation: Literal["aw", "mos", "union"] = "aw"

    # Step A — deskew filter.
    filter_nonfinite_points: bool = True

    # Step A — rolling-shutter motion compensation. When raw NPZ has no
    # per-point timestamps (t_offset_us), this synthesizes them from each
    # point's azimuth assuming uniform rotation. Disabling = all points
    # share the header pose → intra-sweep smear that spreads statics across
    # voxels and leaks them into dynamic_map.
    synthesize_per_point_times: bool = True
    # Rotation period and spin direction come from each lidar's sensor
    # profile (sweep_duration_ms / rotation_dir) — they are datasheet facts.

    # Strictness flags — fail loudly on missing inputs rather than degrade.
    require_patchwork: bool = True
    allow_uncompensated_motion: bool = False

    # Step B — voxel classification.
    voxel_size_m: float = 0.15

    # The one evidence gate: how many ray traversals a voxel needs before its
    # occupancy probability is trusted at all. Statistical, not physical — it
    # trades recall on sparsely-seen structure against noise from single
    # observations. Voxels below it are UNDER_EVIDENCED: neither static nor
    # dynamic. A voxel with zero endpoint hits is FREE_ONLY and can never be
    # dynamic regardless (that rule needs no threshold: "was anything ever
    # measured here?" is a yes/no question).
    min_observations: int = 3

    # Points within this horizontal range of the sensor are never labelled
    # dynamic. Inside a few metres the return is dominated by the ego's own
    # body and near clutter, and free-space carving is maximal, so no method
    # can reliably call motion there. Applies to every segmentation method
    # (aw, mos, union). 0.0 disables.
    dynamic_min_range_m: float = 2.5

    cache_world_xyz_in_memory: bool = True

    # Step D — global static map reduce. The two-pass prior's KDTree match
    # radius is this same value: reduce snaps map points to voxel centres, so
    # "within one map voxel" is exactly what a match means. Its strength and
    # range weighting are derived from the sensor model.
    global_map_voxel_size_m: float = 0.30

    # Unit of ingest's t_offset_us field.
    # Options: "seconds" | "microseconds" | "nanoseconds"
    point_time_unit: str = "seconds"

    # Step C — Patchwork++ parameters.
    patchwork: PatchworkParams = PatchworkParams()

    # Multi-lidar frame grouping (SAM4D alignment).
    frame_sync: FrameSyncParams = FrameSyncParams()

    # MF-MOS parameters (used when segmentation is "mos" or "union").
    mf_mos: MFMosParams = MFMosParams()

    # Fusion parameters (used only when segmentation == "union").
    union: UnionParams = UnionParams()

    # voxel_occupancy.npz alongside static_map.npz. Includes ALL occupied
    # voxels (static + dynamic) — that's what MinkUNet consumes.
    save_voxel_occupancy: bool = True

    # voxel_diag.npz with per-voxel log_odds/n_obs/n_hits/classification for
    # EVERY classified voxel, including carved (log_odds < 0) ones that
    # voxel_occupancy.npz filters out. Powers viz's p_occ color mode.
    save_voxel_diagnostics: bool = False

    # One voxel_occupancy_frame_NNNN.npz per frame_id.
    save_per_frame_voxel_occupancy: bool = False

    @field_validator("voxel_size_m", "global_map_voxel_size_m")
    @classmethod
    def _positive_voxel(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @field_validator("min_observations")
    @classmethod
    def _positive_min_obs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_observations must be >= 1, got {v}")
        return v

    @field_validator("dynamic_min_range_m")
    @classmethod
    def _nonneg_dyn_range(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"dynamic_min_range_m must be >= 0, got {v}")
        return v

    def point_time_scale_to_ns(self) -> float:
        """Multiplier to convert t_offset_us values to nanoseconds."""
        scales = {"seconds": 1e9, "microseconds": 1e3, "nanoseconds": 1.0}
        if self.point_time_unit not in scales:
            raise ValueError(
                f"point_time_unit must be one of {list(scales)}, got {self.point_time_unit!r}"
            )
        return scales[self.point_time_unit]

    def build_sensor_model(self, lidar_id: Optional[str] = None) -> SensorModel:
        """The datasheet SensorModel for ``lidar_id`` (default: the rig's)."""
        return self.sensor_model.build(lidar_id)


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("lidar_preprocessing", {})
    return ComponentConfig.model_validate(section)
