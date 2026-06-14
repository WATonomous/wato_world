"""Pydantic-loaded config for lidar_preprocessing."""

from __future__ import annotations

from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    Used only by the `mos` segmentation method (selected with `--seg mos`).
    The MF-MOS method is fully self-contained: it runs inference (Step A.5)
    and derives static/dynamic clouds purely from the per-sweep moving masks,
    with no Amanatides-Woo / log-odds involvement. Requires a CUDA GPU for
    realistic data; device="cpu" is for tiny smoke tests only.
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str = "/data/models/mf_mos/mf_mos_semantic_kitti.pt"
    arch_config: str = "/data/models/mf_mos/arch_cfg.yaml"
    data_config: str = "/data/models/mf_mos/data_cfg.yaml"
    residual_steps: list[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    range_image_h: int = 32
    range_image_w: int = 1024
    fov_up_deg: float = 10.0
    fov_down_deg: float = -30.0
    device: str = "cuda"
    score_threshold: float = 0.5
    save_scores: bool = False
    # Max pose-interpolation gap (ms) used when warping a historical sweep
    # into the current frame for a residual. Gates POSE quality only.
    max_pose_gap_ms: float = 200.0
    # Max sweep-to-sweep time baseline (ms) accepted for a residual channel.
    # This is distinct from max_pose_gap_ms: a residual at step k spans
    # k * sweep_dt, which for the longer residual_steps legitimately exceeds
    # the pose-gap cap. Conflating the two (the historical bug) zeroed every
    # residual channel whose baseline exceeded max_pose_gap_ms, silently
    # gutting multi-frame MOS. Set comfortably above max(residual_steps) *
    # sweep_dt (e.g. 16 steps * 50 ms @ 20 Hz = 800 ms).
    max_residual_gap_ms: float = 1000.0
    # Match training preprocessing (data_preparing.yaml).
    min_range_m: float = 2.0
    max_range_m: float = 50.0
    # Divisor that scales raw intensity into [0, 1] like KITTI remission.
    # NuScenes intensity is uint8 [0, 255] → 255.0. Use 1.0 for sensors
    # that already produce [0, 1].
    intensity_scale: float = 255.0
    # Restrict MF-MOS to specific lidar_ids. fov_up/down + image H/W are
    # global, so LiDARs with significantly different mount geometry would
    # project into non-KITTI-like range images and the model mispredicts.
    # None = run on every LiDAR; e.g. ["lidar_cc"] = centre only.
    lidar_id_allowlist: Optional[list[str]] = None

    @field_validator("residual_steps")
    @classmethod
    def _residuals_positive(cls, v: list[int]) -> list[int]:
        if any(k <= 0 for k in v):
            raise ValueError(f"residual_steps must all be > 0, got {v}")
        if len(v) != len(set(v)):
            raise ValueError(f"residual_steps must be unique, got {v}")
        return sorted(v)

    @field_validator("score_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score_threshold must be in [0, 1], got {v}")
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
    # share the header pose → ~25cm smear at 7 m/s ego on a 50ms sweep,
    # which spreads statics across voxels and leaks them into dynamic_map.
    synthesize_per_point_times: bool = True
    # Velodyne HDL-32E @ 10 Hz = 100 ms; NuScenes uses same hardware @ 20 Hz = 50 ms.
    lidar_sweep_duration_ms: float = 100.0
    # "auto" detects from data by comparing phi[0] to phi[~200]. NuScenes
    # LIDAR_TOP scans CW (counter to the common "Velodyne is CCW" assumption).
    lidar_rotation_dir: Literal["ccw", "cw", "auto"] = "auto"

    # Strictness flags — defaults make the pipeline fail loudly on missing
    # inputs rather than silently degrade.
    #
    # require_patchwork=True → deskew raises if pypatchworkpp is missing.
    # Without it, ground points pollute static_map.npz AND dynamic_map.npz.
    require_patchwork: bool = True
    #
    # allow_uncompensated_motion=False → deskew raises when a sweep has no
    # per-point timestamps and synthesize_per_point_times is also False.
    allow_uncompensated_motion: bool = False

    # Step B — voxel classification.
    voxel_size_m: float = 0.15
    classification_method: Literal["log_odds", "persistence"] = "log_odds"

    # Persistence-counting parameters (used when classification_method="persistence").
    static_sweep_fraction: float = 0.3
    static_sweep_min: int = 5

    # Log-odds ray-casting parameters (used when classification_method="log_odds").
    l_occ: float = 0.85
    l_free: float = 0.40
    # Must be MUCH larger than l_occ so well-observed statics can build a
    # margin against carving — at clamp=5.0 with l_occ=1.20, the +5 ceiling
    # is reached after ~4 hits and additional hits are wasted while carves
    # keep subtracting. 50.0 gives ~40+ hits of headroom.
    log_odds_clamp: float = 50.0
    p_static_threshold: float = 0.7
    p_dynamic_threshold: float = 0.3
    min_observations: int = 3
    # Voxels with fewer endpoint hits go to free_only (never dynamic).
    min_occupied_hits: int = 1
    # Occupancy-ratio gate for the DYNAMIC class. A voxel may be labelled
    # dynamic only if it was actually occupied on at least this fraction of
    # the rays that traversed it (n_hits / n_obs). Near the ego, scan-plane
    # voxels are pierced by thousands of through-rays (n_obs ~ 1e4) while
    # catching a handful of stray returns (ego self-returns, near clutter);
    # raw p_occ drives them to "dynamic" even though they are free space.
    # True movers are hit on a meaningful fraction of in-view passes (~0.2-0.6)
    # vs ~0.01-0.04 for that near-ego shell, so this cleanly separates them.
    # 0.0 disables the gate (legacy behaviour). 0.10 is a conservative default.
    min_hit_fraction_dynamic: float = 0.10
    # Points within this horizontal range of the sensor are never labelled
    # dynamic. Inside a few metres the return is dominated by the ego's own
    # body and near clutter, and free-space carving is maximal, so no method
    # can reliably call motion there. Applies to every segmentation method
    # (aw, mos, union). 0.0 disables. Mirrors patchwork.min_range and
    # mf_mos.min_range_m.
    dynamic_min_range_m: float = 2.5
    # Default 200 m pairs with range-credibility weighting. Without that
    # weighting, long rays still contribute full weight — set to 50-80 m
    # to recover the historical "drop distant evidence" behaviour.
    max_ray_length_m: float = 200.0
    free_space_margin_voxels: float = 1.0
    # UniLiPs r*_j extension. When True, every log-odds update is scaled by
    # min(1, r_max_credibility_m / d), where d = ray length for the endpoint
    # update and d = parametric entry distance for free-space carving.
    # Down-weights endpoints AND distant carving while preserving near-field
    # carving on long rays.
    use_range_weighted_log_odds: bool = False
    r_max_credibility_m: float = 200.0
    # "skip_endpoint" → traverse ground rays for free-space evidence, no l_occ at endpoint.
    # "skip_ray"      → skip ground rays entirely (legacy).
    ground_endpoint_strategy: Literal["skip_endpoint", "skip_ray"] = "skip_endpoint"

    cache_world_xyz_in_memory: bool = True

    # Step D — global static map reduce.
    global_map_voxel_size_m: float = 0.30

    # Two-pass global map prior (UniLiPs IWU). --two-pass runs single-pass,
    # then reduce, then re-classifies every chunk using global_static_map.npz
    # as a per-sweep KDTree prior.
    #
    # Match radius for the KDTree query. Should be >= global_map_voxel_size_m
    # because reduce snaps points to voxel centres (worst-case offset is
    # voxel_size × √3/2 ≈ 0.26 m at 0.30 m voxel). Raising the global map
    # voxel size without also raising this silently breaks boundary matches.
    global_map_match_radius_m: float = 0.30
    # Per-sweep log-odds boost for map-matched endpoints. Accumulates across
    # every sweep that hits the voxel, so total ≈ l_occ_global_map * r_star
    # * n_sweeps_hitting. Keep at ~1-5% of l_occ so the prior nudges without
    # overriding real evidence. At 0.03, a voxel seen in every sweep of a
    # 30-second 20 Hz chunk accumulates ~0.9 boost.
    l_occ_global_map: float = 0.03

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

    # One voxel_occupancy_frame_NNNN.npz per frame_id (all sweeps sharing
    # that frame_id merged). What perception_2d feeds to MinkUNet — the
    # per-chunk file is only useful for QA.
    save_per_frame_voxel_occupancy: bool = False

    @field_validator("voxel_size_m", "global_map_voxel_size_m", "max_ray_length_m")
    @classmethod
    def _positive_voxel(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @field_validator("l_occ", "l_free")
    @classmethod
    def _positive_log_odds_param(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"log-odds weight must be > 0, got {v}")
        return v

    @field_validator("r_max_credibility_m")
    @classmethod
    def _positive_r_max(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"r_max_credibility_m must be > 0, got {v}")
        return v

    @field_validator("p_static_threshold", "p_dynamic_threshold")
    @classmethod
    def _probability_range(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError(f"probability threshold must be in (0, 1), got {v}")
        return v

    @field_validator("min_observations", "min_occupied_hits")
    @classmethod
    def _positive_min_obs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"value must be >= 1, got {v}")
        return v

    @field_validator("min_hit_fraction_dynamic")
    @classmethod
    def _hit_fraction_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"min_hit_fraction_dynamic must be in [0, 1], got {v}")
        return v

    @field_validator("dynamic_min_range_m")
    @classmethod
    def _nonneg_dyn_range(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"dynamic_min_range_m must be >= 0, got {v}")
        return v

    @field_validator("static_sweep_fraction")
    @classmethod
    def _fraction_range(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError(f"static_sweep_fraction must be in (0, 1], got {v}")
        return v

    @field_validator("static_sweep_min")
    @classmethod
    def _positive_min(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"static_sweep_min must be >= 1, got {v}")
        return v

    def point_time_scale_to_ns(self) -> float:
        """Multiplier to convert t_offset_us values to nanoseconds."""
        scales = {"seconds": 1e9, "microseconds": 1e3, "nanoseconds": 1.0}
        if self.point_time_unit not in scales:
            raise ValueError(
                f"point_time_unit must be one of {list(scales)}, got {self.point_time_unit!r}"
            )
        return scales[self.point_time_unit]


def load_config(path: str) -> ComponentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    section = data.get("lidar_preprocessing", {})
    return ComponentConfig.model_validate(section)
