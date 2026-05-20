"""Step A2 — MapMOS chunk orchestrator.

Loads the MinkUNet model once per chunk via `weights.load_and_validate`,
builds a running `MapAccumulator`, iterates sweeps in order, runs
inference via `run_sweep_inference`, writes the per-sweep sidecar
logit file, and registers the scan into the accumulator AFTER the
sidecar is written (plan non-negotiable #30 — adding before would let
the model see itself in the map).

Per-sweep failures are isolated: a warning is logged, no sidecar
written, and classify falls through to geometry-only for that sweep.
CUDA OOM has a dedicated branch (plan non-negotiable #16) — without
empty_cache, one OOM cascades into silent OOMs for every subsequent
sweep in the chunk.

When model loading fails (e.g., missing checkpoint, torch not
installed, MinkowskiEngine wasn't compiled into the image), the chunk
falls back to the zero-stub path — every sidecar is length-N zeros,
preserving the Step-1 regression invariant. This keeps the geometry
path runnable even if the MapMOS image hasn't been fully built yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from wato_common.artifact_store import (
    ensure_local_dir,
    lidar_proc_dir,
    lidar_proc_index_path,
    mapmos_logit_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import PROCESSED_SWEEPS_SCHEMA, ProcessedSweepMeta
from wato_lidar_preprocessing.classify.io_helpers import load_world_full
from wato_lidar_preprocessing.config import ComponentConfig

from .inference import run_sweep_inference
from .io import write_logits

log = logging.getLogger(__name__)


@dataclass
class MapMOSResult:
    n_sweeps_processed: int
    n_sweeps_skipped: int
    n_sweeps_failed: int


def process_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    prev_chunk_id: str | None = None,  # noqa: ARG001 — kept for future cross-chunk warm history (Phase 5)
) -> MapMOSResult:
    """Write MapMOS logit sidecars for every valid sweep in the chunk.

    Side effects:
      - One <sweep_id>_mapmos_logit.npy per valid sweep (or none on
        per-sweep failure).
      - lidar_proc_index.parquet rewritten with mapmos_logit_path column
        populated for sweeps whose sidecar was written.
    """
    if not cfg.mapmos.enabled:
        log.debug("chunk %s: mapmos.enabled=False — skipping inference", chunk_id)
        return MapMOSResult(0, 0, 0)

    meta_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    if not meta_rows:
        log.warning("chunk %s: lidar_proc_index empty — nothing to infer", chunk_id)
        return MapMOSResult(0, 0, 0)

    ensure_local_dir(lidar_proc_dir(bag_id, chunk_id))

    # --- Load model + build accumulator ---------------------------------
    # Both can fail when torch / MinkowskiEngine / mapmos aren't installed
    # (image not yet built with the MapMOS deps). On failure we fall back
    # to the zero-stub path so the chunk still produces aligned sidecars
    # and the geometry-only regression contract holds.
    model, ckpt_voxel_size, device, oom_exc_class = _try_load_model(
        cfg.mapmos.weights_path, cfg.mapmos.device, chunk_id
    )
    accumulator = _try_build_accumulator(chunk_id)

    n_ok = 0
    n_skipped = 0
    n_failed = 0
    updated_rows: list[dict] = []

    for row in tqdm(
        meta_rows,
        desc=f"mapmos chunk {chunk_id}",
        unit="sweep",
    ):
        if row.get("valid") is False:
            # No sidecar — classify falls through to geometry-only for
            # this sweep. Plan non-negotiable #7.
            n_skipped += 1
            updated_rows.append(_passthrough_row(row, mapmos_path=None))
            continue

        try:
            mapmos_path_uri = _process_one_sweep(
                row,
                cfg=cfg,
                bag_id=bag_id,
                chunk_id=chunk_id,
                model=model,
                ckpt_voxel_size=ckpt_voxel_size,
                device=device,
                accumulator=accumulator,
            )
            n_ok += 1
            updated_rows.append(_passthrough_row(row, mapmos_path=mapmos_path_uri))
        except oom_exc_class as exc:
            # Plan non-negotiable #16: empty_cache or every subsequent
            # sweep in the chunk OOMs silently with no sidecar.
            log.warning(
                "chunk %s sweep %s: CUDA OOM — skipping MapMOS for this sweep (%s)",
                chunk_id,
                row.get("sweep_id"),
                exc,
            )
            _try_empty_cuda_cache()
            n_failed += 1
            updated_rows.append(_passthrough_row(row, mapmos_path=None))
        except Exception as exc:  # noqa: BLE001 — isolate per-sweep failures
            log.warning(
                "chunk %s sweep %s: MapMOS inference failed (%s) — geometry-only for this sweep",
                chunk_id,
                row.get("sweep_id"),
                exc,
            )
            n_failed += 1
            updated_rows.append(_passthrough_row(row, mapmos_path=None))

    # Persist the index with the new column populated. write_table validates
    # against PROCESSED_SWEEPS_SCHEMA so the parquet stays self-describing.
    write_table(
        updated_rows,
        PROCESSED_SWEEPS_SCHEMA,
        lidar_proc_index_path(bag_id, chunk_id),
    )

    log.info(
        "chunk %s mapmos: ok=%d skipped(valid=False)=%d failed=%d",
        chunk_id,
        n_ok,
        n_skipped,
        n_failed,
    )
    return MapMOSResult(
        n_sweeps_processed=n_ok,
        n_sweeps_skipped=n_skipped,
        n_sweeps_failed=n_failed,
    )


# ---------------------------------------------------------------------------
# Lazy resource setup — model + accumulator + OOM exception class
# ---------------------------------------------------------------------------
class _NeverRaised(Exception):
    """Sentinel OOM class for the stub-fallback path.

    When torch isn't importable we still need *something* to put in the
    `except (...)` clause. Using a class that nothing in the codebase
    will ever raise makes that branch effectively dead in stub mode.
    """


def _try_load_model(weights_path: str, device_str: str, chunk_id: str):
    """Return (model, ckpt_voxel_size, device, oom_exc_class).

    On any failure (torch missing, MinkowskiEngine missing, mapmos package
    missing, checkpoint file absent, version mismatch) returns the stub
    quad `(None, None, None, _NeverRaised)`. The chunk then runs in
    zero-prior stub mode — every sidecar gets length-N zeros, preserving
    the Step-1 regression invariant.
    """
    try:
        import torch

        from .weights import load_and_validate

        # Resolve device: cuda → fall back to cpu with warning if unavailable.
        if device_str == "cuda" and not torch.cuda.is_available():
            log.warning(
                "chunk %s: cfg.mapmos.device='cuda' but no CUDA device "
                "visible — falling back to cpu",
                chunk_id,
            )
            device_str = "cpu"
        device = torch.device(device_str)

        model, ckpt_voxel_size = load_and_validate(weights_path, device)
        oom_exc_class = (torch.cuda.OutOfMemoryError, MemoryError)
        log.info(
            "chunk %s: loaded MapMOS weights from %s (voxel_size=%.3fm, device=%s)",
            chunk_id,
            weights_path,
            ckpt_voxel_size,
            device,
        )
        return model, ckpt_voxel_size, device, oom_exc_class
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        log.warning(
            "chunk %s: MapMOS model load failed (%s) — falling back to "
            "zero-stub inference (sidecars will be length-N zeros)",
            chunk_id,
            exc,
        )
        return None, None, None, _NeverRaised


def _try_build_accumulator(chunk_id: str):
    """Construct the MapAccumulator, or None if the pybind binding isn't installed."""
    try:
        from .map_accumulator import MapAccumulator

        return MapAccumulator()
    except (ImportError, ModuleNotFoundError) as exc:
        log.warning(
            "chunk %s: MapAccumulator unavailable (%s) — running inference "
            "without an accumulator (every sweep sees an empty 'map')",
            chunk_id,
            exc,
        )
        return None


def _try_empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — defensive only
        pass


# ---------------------------------------------------------------------------
# Per-sweep work
# ---------------------------------------------------------------------------
def _process_one_sweep(
    row: dict,
    *,
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    model,
    ckpt_voxel_size,
    device,
    accumulator,
) -> str:
    """Run inference for one sweep, write the sidecar, register into accumulator.

    Returns the URI of the written sidecar. Raises on inference failure
    (caller's per-sweep try/except handles it).

    Plan non-negotiable #30: `accumulator.add_scan(...)` runs AFTER the
    sidecar is written. The model's "map" input must contain only PRIOR
    scans — adding the current scan first would let the model see itself.
    """
    sweep_id = int(row["sweep_id"])
    world_uri = row["world_path"]

    # Length invariant is asserted against the ACTUAL xyz length from the
    # NPZ, NOT the parquet n_points_total column (plan non-negotiable #14).
    xyz_world, _intensity, sensor_origin, ground_mask = load_world_full(world_uri)
    n_total = int(xyz_world.shape[0])

    if n_total == 0:
        # Legitimate empty sweep: still write a zero-length sidecar so
        # classify sees explicit alignment. Plan non-negotiable #20.
        out_uri = mapmos_logit_path(bag_id, chunk_id, sweep_id)
        write_logits(out_uri, np.empty(0, dtype=np.float32))
        return out_uri

    # Ego world position comes from the deskew NPZ's `origin` field
    # (sensor pose's translation in world frame at the sweep's reference
    # timestamp). Required for the range filter relative to ego when
    # we're running real inference; in stub-mode (model is None) we
    # don't use it, so a missing origin is harmless.
    if sensor_origin is None and model is not None:
        log.warning(
            "sweep %s: world NPZ missing 'origin' field — falling back to "
            "stub for this sweep (re-run deskew to fix)",
            sweep_id,
        )

    logits = run_sweep_inference(
        xyz_world=xyz_world,
        ego_world_xyz=sensor_origin,  # None → stub fallback inside run_sweep_inference
        ground_mask=ground_mask,
        scan_index=sweep_id,
        map_accumulator=accumulator,
        model=model,
        ckpt_voxel_size=ckpt_voxel_size,
        device=device,
        min_range_m=cfg.mapmos.min_range_m,
        max_range_m=cfg.mapmos.max_range_m,
        logit_clamp=cfg.mapmos.fusion.logit_clamp,
    )

    # Length invariant — assertion uses xyz.shape[0], NOT n_points_total
    # (plan non-negotiable #14).
    if logits.shape[0] != n_total:
        raise RuntimeError(
            f"sweep {sweep_id}: run_sweep_inference returned length "
            f"{logits.shape[0]}, expected {n_total} (xyz NPZ length)"
        )
    if logits.dtype != np.float32:
        logits = logits.astype(np.float32)

    out_uri = mapmos_logit_path(bag_id, chunk_id, sweep_id)
    write_logits(out_uri, logits)

    # Register THIS scan into the accumulator AFTER the sidecar is written
    # (plan non-negotiable #30). Future sweeps see this scan's points as
    # "map" with the current scan_index attached. Skipped in stub-mode
    # (model is None — no inference happened, no point growing the map)
    # and when sensor_origin is missing (VHM needs ego for pruning).
    if accumulator is not None and model is not None and sensor_origin is not None:
        try:
            accumulator.add_scan(xyz_world, sensor_origin, sweep_id)
        except Exception as exc:  # noqa: BLE001 — accumulator failure must not poison the chunk
            log.warning(
                "chunk %s sweep %s: accumulator.add_scan failed (%s) — "
                "future sweeps in this chunk will see a smaller map",
                chunk_id,
                sweep_id,
                exc,
            )

    return out_uri


def _passthrough_row(row: dict, *, mapmos_path: str | None) -> dict:
    """Round-trip a meta row through ProcessedSweepMeta, setting mapmos_logit_path.

    Going through the pydantic model + .model_dump() guarantees the row
    has every column PROCESSED_SWEEPS_SCHEMA expects (including the new
    mapmos_logit_path column with its default None) even when the input
    row came from an older parquet that pre-dates the schema bump.
    """
    return ProcessedSweepMeta(
        bag_id=row["bag_id"],
        chunk_id=row["chunk_id"],
        sweep_id=int(row["sweep_id"]),
        lidar_id=row["lidar_id"],
        reference_timestamp_ns=int(row["reference_timestamp_ns"]),
        n_points_total=int(row.get("n_points_total") or 0),
        n_points_static=int(row.get("n_points_static") or 0),
        n_points_dynamic=int(row.get("n_points_dynamic") or 0),
        n_points_ground=int(row.get("n_points_ground") or 0),
        world_path=row.get("world_path", ""),
        dynamic_mask_path=row.get("dynamic_mask_path", "") or "",
        mapmos_logit_path=mapmos_path,
        has_intensity=bool(row.get("has_intensity", False)),
        deskewed=bool(row.get("deskewed", False)),
        valid=bool(row.get("valid", True)),
        drop_reason=row.get("drop_reason"),
        world_xmin=row.get("world_xmin"),
        world_xmax=row.get("world_xmax"),
        world_ymin=row.get("world_ymin"),
        world_ymax=row.get("world_ymax"),
        world_zmin=row.get("world_zmin"),
        world_zmax=row.get("world_zmax"),
        frame_id=row.get("frame_id"),
    ).model_dump()
