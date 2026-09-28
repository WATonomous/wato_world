"""lidar_preprocessing pipeline orchestrator.

Runs Steps A → A.5 → B → C for each chunk:
  A.   deskew   — motion compensation + world-frame projection (+ Patchwork++)
  A.5. mf_mos   — learned moving-object segmentation (no-op when disabled)
  B.   classify — voxel static/dynamic decomposition
  C.   ground   — per-sweep ground aggregation + height grid

Full-bag runs publish Step D reductions. In two-pass mode, run() builds a
staged prior and reruns classify, ground, and summary before publication.

Outputs are written to a run-scoped staging tree and promoted only after all
selected chunks and required reductions validate.  Completion records, rather
than the presence of any individual artifact, provide idempotency.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import shutil
import tempfile
import traceback
import uuid
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from wato_common.artifact_store import (
    chunks_index_path,
    dynamic_map_path,
    global_ground_path,
    global_static_map_path,
    ground_path,
    lidar_bag_manifest_path,
    lidar_chunk_root,
    lidar_completion_path,
    lidar_preprocessing_root,
    lidar_proc_index_path,
    lidar_proc_summary_path,
    lidar_sweeps_path,
    local_path,
    poses_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import CHUNK_SUMMARY_SCHEMA, ChunkSummaryRow
from wato_lidar_preprocessing import classify, deskew, ground, mf_mos as mf_mos_step
from wato_lidar_preprocessing.config import ComponentConfig
from wato_common.storage import component_version
from wato_lidar_preprocessing.reduce import reduce_ground_map, reduce_static_map

log = logging.getLogger(__name__)


def _write_chunk_summary(
    bag_id: str,
    chunk_id: str,
    classify_result: classify.ClassifyResult,
    ground_result: ground.GroundResult,
    mf_mos_result: mf_mos_step.MFMosResult | None = None,
) -> None:
    """Aggregate per-sweep parquet + classify/ground results into one row.

    Per-sweep counts come from lidar_proc_index (after classify updated it).
    Runtime stats come from the step result objects.
    """
    sweep_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    n_total = len(sweep_rows)
    n_invalid = sum(1 for r in sweep_rows if r.get("valid") is False)
    n_valid = n_total - n_invalid
    n_points_total = sum(int(r.get("n_points_total") or 0) for r in sweep_rows)
    n_points_static = sum(int(r.get("n_points_static") or 0) for r in sweep_rows)
    n_points_dynamic = sum(int(r.get("n_points_dynamic") or 0) for r in sweep_rows)
    n_points_ground = sum(int(r.get("n_points_ground") or 0) for r in sweep_rows)

    old_mf: dict = {}
    summary_path = lidar_proc_summary_path(bag_id, chunk_id)
    if mf_mos_result is None and os.path.exists(local_path(summary_path)):
        old_rows = read_rows(summary_path)
        if old_rows:
            old_mf = old_rows[0]

    summary = ChunkSummaryRow(
        bag_id=bag_id,
        chunk_id=chunk_id,
        n_sweeps_total=n_total,
        n_sweeps_valid=n_valid,
        n_sweeps_invalid=n_invalid,
        n_points_total=n_points_total,
        n_points_static=n_points_static,
        n_points_dynamic=n_points_dynamic,
        n_points_ground=n_points_ground,
        n_rejected_dynamic_ground=ground_result.n_rejected_dynamic_ground,
        cache_auto_disabled=classify_result.cache_auto_disabled,
        estimated_cache_bytes=classify_result.estimated_cache_bytes,
        ground_status=ground_result.status,
        mf_mos_n_processed=(
            mf_mos_result.n_sweeps_processed
            if mf_mos_result
            else old_mf.get("mf_mos_n_processed")
        ),
        mf_mos_n_skipped=(
            mf_mos_result.n_skipped
            if mf_mos_result
            else old_mf.get("mf_mos_n_skipped")
        ),
        mf_mos_n_points_moving=(
            mf_mos_result.n_points_moving
            if mf_mos_result
            else old_mf.get("mf_mos_n_points_moving")
        ),
    )
    write_table(
        [summary.model_dump()],
        CHUNK_SUMMARY_SCHEMA,
        summary_path,
    )


def _config_digest(cfg: ComponentConfig) -> str:
    payload = json.dumps(
        cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expected_sweep_count(bag_id: str, chunk_id: str) -> int:
    rows = read_rows(lidar_sweeps_path(bag_id, chunk_id))
    return len(
        {
            (str(r["lidar_id"]), int(r["sweep_id"]))
            for r in rows
            if r.get("valid", True)
        }
    )


def _chunk_complete(
    cfg: ComponentConfig, bag_id: str, chunk_id: str, mode: str
) -> bool:
    """Return true only for a matching, complete v2 publication record."""
    path = local_path(lidar_completion_path(bag_id, chunk_id))
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
        expected = _expected_sweep_count(bag_id, chunk_id)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return record == {
        "artifact_version": component_version("lidar_preprocessing"),
        "mode": mode,
        "configuration_digest": _config_digest(cfg),
        "run_id": record.get("run_id"),
        "expected_composite_sweeps": expected,
        "completed_composite_sweeps": expected,
    } and isinstance(record.get("run_id"), str) and bool(record["run_id"])


@contextmanager
def _staged_lidar_outputs(stage_root: str):
    previous = os.environ.get("WATO_LIDAR_STAGING_ROOT")
    os.environ["WATO_LIDAR_STAGING_ROOT"] = stage_root
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WATO_LIDAR_STAGING_ROOT", None)
        else:
            os.environ["WATO_LIDAR_STAGING_ROOT"] = previous


def _validate_staged_chunk(bag_id: str, chunk_id: str) -> int:
    required = (
        lidar_proc_index_path(bag_id, chunk_id),
        lidar_proc_summary_path(bag_id, chunk_id),
        static_map_path(bag_id, chunk_id),
        dynamic_map_path(bag_id, chunk_id),
        ground_path(bag_id, chunk_id),
    )
    missing = [uri for uri in required if not os.path.exists(local_path(uri))]
    if missing:
        raise RuntimeError(f"chunk {chunk_id!r} staged artifacts missing: {missing}")
    expected = _expected_sweep_count(bag_id, chunk_id)
    rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    completed = len({(str(r["lidar_id"]), int(r["sweep_id"])) for r in rows})
    if completed != expected:
        raise RuntimeError(
            f"chunk {chunk_id!r} completed {completed}/{expected} composite sweeps"
        )
    for row in rows:
        if not row.get("valid", True):
            continue
        world_uri = row.get("world_path")
        if not world_uri or not os.path.exists(local_path(world_uri)):
            raise RuntimeError(
                f"chunk {chunk_id!r} world_path missing for "
                f"({row['lidar_id']}, {row['sweep_id']})"
            )
        world = np.load(local_path(world_uri))
        n_points = int(world["x"].shape[0])
        masks: dict[str, np.ndarray] = {}
        for field in ("static_mask_path", "dynamic_mask_path"):
            uri = row.get(field)
            if not uri or not os.path.exists(local_path(uri)):
                raise RuntimeError(
                    f"chunk {chunk_id!r} {field} missing for "
                    f"({row['lidar_id']}, {row['sweep_id']})"
                )
            mask = np.load(local_path(uri))
            if mask.dtype != np.bool_ or mask.shape != (n_points,):
                raise RuntimeError(
                    f"chunk {chunk_id!r} {field} is not bool[{n_points}] for "
                    f"({row['lidar_id']}, {row['sweep_id']})"
                )
            masks[field] = mask
        if np.any(masks["static_mask_path"] & masks["dynamic_mask_path"]):
            raise RuntimeError(
                f"chunk {chunk_id!r} static/dynamic masks overlap for "
                f"({row['lidar_id']}, {row['sweep_id']})"
            )
        if row.get("mf_mos_status") == "ok":
            mf_uri = row.get("mf_mos_mask_path")
            if not mf_uri or not os.path.exists(local_path(mf_uri)):
                raise RuntimeError(
                    f"chunk {chunk_id!r} MF-MOS status ok without a mask for "
                    f"({row['lidar_id']}, {row['sweep_id']})"
                )
    return completed


def _write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
        fh.write("\n")
    os.replace(temp_path, path)


def _validate_chunk_inputs(bag_id: str, chunk_id: str) -> None:
    """Confirm ingest produced the per-chunk artifacts. Raises with a clear
    message rather than letting deskew fail deep inside."""
    required = {
        "lidar_sweeps.parquet": lidar_sweeps_path(bag_id, chunk_id),
        "poses.parquet": poses_path(bag_id, chunk_id),
    }
    missing = [
        name for name, uri in required.items() if not os.path.exists(local_path(uri))
    ]
    if missing:
        raise FileNotFoundError(
            f"chunk {chunk_id!r} of bag {bag_id!r} is missing ingest artifacts "
            f"({', '.join(missing)}); re-run `watod run ingest <bag>`."
        )


def _process_one_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
) -> tuple[str, bool, str]:
    """Run A → A.5 → B → C for a single chunk.

    Returns (chunk_id, ok, error_msg). error_msg carries the full traceback
    so it survives the ProcessPoolExecutor worker boundary.
    """
    try:
        _validate_chunk_inputs(bag_id, chunk_id)

        log.info("=== chunk %s: step A — deskew ===", chunk_id)
        deskew.process_chunk(cfg, bag_id, chunk_id)

        log.info(
            "=== chunk %s: step A.5 — mf_mos (enabled=%s) ===",
            chunk_id,
            cfg.mf_mos.enabled,
        )
        mf_mos_result = mf_mos_step.process_chunk(cfg, bag_id, chunk_id)

        log.info("=== chunk %s: step B — classify ===", chunk_id)
        classify_result = classify.process_chunk(cfg, bag_id, chunk_id)

        log.info("=== chunk %s: step C — ground ===", chunk_id)
        ground_result = ground.process_chunk(cfg, bag_id, chunk_id)

        _write_chunk_summary(
            bag_id, chunk_id, classify_result, ground_result, mf_mos_result
        )
        return (chunk_id, True, "")
    except Exception as exc:  # noqa: BLE001 — one chunk failing must not stop the rest
        log.exception("chunk %s failed", chunk_id)
        tb = traceback.format_exc()
        return (chunk_id, False, f"{type(exc).__name__}: {exc}\n{tb}")


def _pass2_chunk_worker(
    chunk_id: str,
    cfg: ComponentConfig,
    bag_id: str,
    global_map_path: str,
) -> None:
    """ProcessPoolExecutor worker. Must stay module-scope (closures aren't
    picklable). Each worker rebuilds the KDTree from disk rather than pickling
    a large cKDTree across the pool pipe.
    """
    prior = classify.GlobalMapPrior.from_npz(
        global_map_path,
        match_radius_m=cfg.global_map_match_radius_m,
    )
    classify_result = classify.process_chunk(
        cfg, bag_id, chunk_id, global_map_prior=prior
    )
    ground_result = ground.process_chunk(cfg, bag_id, chunk_id)
    _write_chunk_summary(bag_id, chunk_id, classify_result, ground_result)


def _run_classify_pass2(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str | None,
    workers: int,
    global_map_path: str,
) -> None:
    """Re-run classify, ground, and summary with the staged static prior."""
    chunk_rows = read_rows(chunks_index_path(bag_id))
    if chunk_id:
        chunk_rows = [r for r in chunk_rows if r["chunk_id"] == chunk_id]

    n_total = len(chunk_rows)
    n_ok = 0
    failures: list[tuple[str, str]] = []

    if workers <= 1:
        # Build the KDTree once and reuse across chunks in this process.
        prior = classify.GlobalMapPrior.from_npz(
            global_map_path,
            match_radius_m=cfg.global_map_match_radius_m,
        )
        for row in chunk_rows:
            cid = row["chunk_id"]
            try:
                classify_result = classify.process_chunk(
                    cfg, bag_id, cid, global_map_prior=prior
                )
                ground_result = ground.process_chunk(cfg, bag_id, cid)
                _write_chunk_summary(
                    bag_id, cid, classify_result, ground_result
                )
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 — one chunk failing must not stop the rest
                log.exception("pass 2 chunk %s failed", cid)
                failures.append(
                    (cid, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
                )
    else:
        # Each worker rebuilds its own KDTree; pickling cKDTree across the
        # pool pipe per chunk is slower than rebuilding from disk.
        worker = functools.partial(
            _pass2_chunk_worker,
            cfg=cfg,
            bag_id=bag_id,
            global_map_path=global_map_path,
        )
        log.info("pass 2: running %d chunks across %d workers", n_total, workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(worker, r["chunk_id"]): r["chunk_id"] for r in chunk_rows
            }
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    fut.result()
                    n_ok += 1
                except Exception as exc:  # noqa: BLE001
                    log.exception("pass 2 chunk %s failed", cid)
                    failures.append((cid, f"{type(exc).__name__}: {exc}"))

    log.info(
        "two-pass mode: pass 2 complete for bag %s: %d/%d chunks succeeded",
        bag_id,
        n_ok,
        n_total,
    )
    if failures:
        details = "\n".join(f"--- chunk {cid} ---\n{err}" for cid, err in failures)
        raise RuntimeError(
            f"lidar_preprocessing pass 2: {len(failures)}/{n_total} chunks failed\n"
            f"{details}"
        )


def _promote_staged_run(
    *,
    cfg: ComponentConfig,
    bag_id: str,
    chunk_ids: list[str],
    stage_root: str,
    mode: str,
    run_id: str,
    publish_bag_outputs: bool,
) -> None:
    """Promote validated output, then publish completion records last."""
    canonical_bag = local_path(lidar_preprocessing_root(bag_id))
    staged_bag = os.path.join(stage_root, bag_id)
    backup_root = os.path.join(stage_root, "_backup")
    os.makedirs(canonical_bag, exist_ok=True)
    os.makedirs(backup_root, exist_ok=True)
    promoted: list[tuple[str, str | None]] = []
    global_promoted: list[tuple[str, str | None]] = []
    try:
        for cid in chunk_ids:
            source = os.path.join(staged_bag, cid)
            target = local_path(lidar_chunk_root(bag_id, cid))
            backup = os.path.join(backup_root, cid)
            old_backup: str | None = None
            if os.path.exists(target):
                os.replace(target, backup)
                old_backup = backup
            os.replace(source, target)
            promoted.append((target, old_backup))

        digest = _config_digest(cfg)
        for cid in chunk_ids:
            expected = _expected_sweep_count(bag_id, cid)
            _write_json_atomic(
                local_path(lidar_completion_path(bag_id, cid)),
                {
                    "artifact_version": component_version("lidar_preprocessing"),
                    "mode": mode,
                    "configuration_digest": digest,
                    "run_id": run_id,
                    "expected_composite_sweeps": expected,
                    "completed_composite_sweeps": expected,
                },
            )

        if publish_bag_outputs:
            for uri in (global_static_map_path(bag_id), global_ground_path(bag_id)):
                target = local_path(uri)
                source = os.path.join(staged_bag, os.path.basename(target))
                backup = os.path.join(backup_root, os.path.basename(target))
                old_backup = None
                if os.path.exists(target):
                    os.replace(target, backup)
                    old_backup = backup
                os.replace(source, target)
                global_promoted.append((target, old_backup))
            _write_json_atomic(
                local_path(lidar_bag_manifest_path(bag_id)),
                {
                    "artifact_version": component_version("lidar_preprocessing"),
                    "mode": mode,
                    "configuration_digest": digest,
                    "run_id": run_id,
                    "chunks": chunk_ids,
                },
            )
    except Exception:
        for target, backup in reversed(global_promoted):
            if os.path.exists(target):
                os.remove(target)
            if backup and os.path.exists(backup):
                os.replace(backup, target)
        for target, backup in reversed(promoted):
            if os.path.isdir(target):
                shutil.rmtree(target)
            if backup and os.path.exists(backup):
                os.replace(backup, target)
        raise


def run(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: str | None = None,
    force: bool = False,
    workers: int = 1,
    two_pass: bool = True,
) -> None:
    """Process all chunks (or one) for a bag.

    Args:
        cfg: parsed ComponentConfig.
        bag_id: bag identifier (must have been ingested).
        chunk_id: optional single-chunk filter.
        force: re-process chunks whose ground.npz already exists.
        workers: concurrent worker processes (>=1).
        two_pass: when True (default), after pass 1 builds the bag-level
            global_static_map.npz via reduce_static_map and re-runs classify
            on every chunk using that map as a per-sweep KDTree prior
            (UniLiPs IWU). Roughly doubles wall time; improves static recall
            on long-range structure sparsely observed in any one chunk.
    """
    chunks_idx = chunks_index_path(bag_id)
    if not os.path.exists(local_path(chunks_idx)):
        raise FileNotFoundError(
            f"chunks index not found for bag {bag_id!r} at {chunks_idx} — "
            "run `watod run ingest <bag>` first."
        )

    chunk_rows = read_rows(chunks_idx)
    if chunk_id:
        chunk_rows = [r for r in chunk_rows if r["chunk_id"] == chunk_id]
        if not chunk_rows:
            raise ValueError(f"chunk_id {chunk_id!r} not found for bag {bag_id!r}")

    mode = (
        "two_pass_local"
        if two_pass and chunk_id is not None
        else "two_pass_global"
        if two_pass
        else "one_pass"
    )
    selected = [str(row["chunk_id"]) for row in chunk_rows]
    complete = [
        cid
        for cid in selected
        if not force and _chunk_complete(cfg, bag_id, cid, mode)
    ]
    if len(complete) == len(selected):
        log.info(
            "lidar_preprocessing: all %d chunks already processed for bag %s",
            len(chunk_rows),
            bag_id,
        )
        return

    # A full-bag publication is one coherent run.  If even one record is stale,
    # stage every bag chunk again so its prior and reductions have one lineage.
    pending = selected if chunk_id is None else [selected[0]]

    n_total = len(pending)
    canonical_bag = local_path(lidar_preprocessing_root(bag_id))
    staging_parent = os.path.join(os.path.dirname(canonical_bag), ".staging")
    os.makedirs(staging_parent, exist_ok=True)
    stage_root = tempfile.mkdtemp(prefix="lidar-run-", dir=staging_parent)
    run_id = uuid.uuid4().hex
    try:
        with _staged_lidar_outputs(stage_root):
            n_ok = 0
            failures: list[tuple[str, str]] = []
            if workers <= 1:
                for cid in pending:
                    _, ok, err = _process_one_chunk(cfg, bag_id, cid)
                    if ok:
                        n_ok += 1
                    else:
                        failures.append((cid, err))
            else:
                log.info("running %d chunks across %d workers", n_total, workers)
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_process_one_chunk, cfg, bag_id, cid): cid
                        for cid in pending
                    }
                    for fut in as_completed(futures):
                        cid, ok, err = fut.result()
                        if ok:
                            n_ok += 1
                        else:
                            failures.append((cid, err))

            if failures:
                details = "\n".join(
                    f"--- chunk {cid} ---\n{err}" for cid, err in failures
                )
                raise RuntimeError(
                    f"lidar_preprocessing: {len(failures)}/{n_total} chunks failed\n"
                    f"{details}"
                )

            if two_pass:
                global_map_uri = reduce_static_map(bag_id, cfg)
                _run_classify_pass2(
                    cfg,
                    bag_id,
                    chunk_id,
                    workers,
                    local_path(global_map_uri),
                )

            for cid in pending:
                _validate_staged_chunk(bag_id, cid)

            publish_bag_outputs = chunk_id is None
            if publish_bag_outputs:
                # Rebuild both reductions from the final (pass-two when used)
                # staged maps.  Nothing bag-global is emitted for local runs.
                reduce_static_map(bag_id, cfg)
                reduce_ground_map(bag_id, cfg)

        _promote_staged_run(
            cfg=cfg,
            bag_id=bag_id,
            chunk_ids=pending,
            stage_root=stage_root,
            mode=mode,
            run_id=run_id,
            publish_bag_outputs=chunk_id is None,
        )
        log.info(
            "lidar_preprocessing published bag %s run %s (%d chunks, mode=%s)",
            bag_id,
            run_id,
            n_ok,
            mode,
        )
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


__all__ = ["run", "_chunk_complete", "_process_one_chunk"]
