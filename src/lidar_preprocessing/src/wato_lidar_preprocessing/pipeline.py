"""lidar_preprocessing pipeline orchestrator.

Runs Steps A → B → C for each chunk in order:
  A. deskew    — per-point motion compensation + world-frame projection
                 (also runs Patchwork++ per sweep in sensor frame)
  B. classify  — voxel static/dynamic decomposition
  C. ground    — aggregates per-sweep ground masks + height grid

Step D (reduce) is a separate post-chunk operation triggered by the
`reduce` CLI subcommand.

Features:
- Upstream validation: rejects bags that haven't been ingested.
- Idempotency: skips chunks whose ground.npz already exists, unless force=True.
- Per-chunk isolation: a failure in chunk N does not stop chunks N+1...
- Parallelism: pass workers > 1 to run chunks concurrently in a ProcessPool.
"""

from __future__ import annotations

import functools
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from wato_common.artifact_store import (
    chunks_index_path,
    ground_path,
    lidar_proc_index_path,
    lidar_proc_summary_path,
    lidar_sweeps_path,
    local_path,
    poses_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import CHUNK_SUMMARY_SCHEMA, ChunkSummaryRow
from wato_lidar_preprocessing import classify, deskew, ground, mf_mos as mf_mos_step
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.reduce import reduce_static_map

log = logging.getLogger(__name__)


def _write_chunk_summary(
    bag_id: str,
    chunk_id: str,
    classify_result: classify.ClassifyResult,
    ground_result: ground.GroundResult,
    mf_mos_result: mf_mos_step.MFMosResult | None = None,
) -> None:
    """Aggregate the per-sweep parquet + classify/ground results into one row.

    The per-sweep counts come from lidar_proc_index (after classify updated
    it).  Runtime stats (cache_auto_disabled, ground status) come from the
    step result objects so we don't have to re-derive them.
    """
    sweep_rows = read_rows(lidar_proc_index_path(bag_id, chunk_id))
    n_total = len(sweep_rows)
    n_invalid = sum(1 for r in sweep_rows if r.get("valid") is False)
    n_valid = n_total - n_invalid
    n_points_total = sum(int(r.get("n_points_total") or 0) for r in sweep_rows)
    n_points_static = sum(int(r.get("n_points_static") or 0) for r in sweep_rows)
    n_points_dynamic = sum(int(r.get("n_points_dynamic") or 0) for r in sweep_rows)
    n_points_ground = sum(int(r.get("n_points_ground") or 0) for r in sweep_rows)

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
        n_dropped_dynamic_ground=ground_result.n_dropped_dynamic,
        cache_auto_disabled=classify_result.cache_auto_disabled,
        estimated_cache_bytes=classify_result.estimated_cache_bytes,
        ground_status=ground_result.status,
        mf_mos_n_processed=mf_mos_result.n_sweeps_processed if mf_mos_result else None,
        mf_mos_n_skipped=mf_mos_result.n_skipped if mf_mos_result else None,
        mf_mos_n_points_moving=mf_mos_result.n_points_moving if mf_mos_result else None,
    )
    write_table(
        [summary.model_dump()],
        CHUNK_SUMMARY_SCHEMA,
        lidar_proc_summary_path(bag_id, chunk_id),
    )


def _chunk_complete(bag_id: str, chunk_id: str) -> bool:
    """A chunk is considered complete when ground.npz exists.

    Both real ground.npz and the sentinel (status='skipped_no_ground_mask')
    count as complete — re-running won't help unless the upstream changes.
    """
    return os.path.exists(local_path(ground_path(bag_id, chunk_id)))


def _validate_chunk_inputs(bag_id: str, chunk_id: str) -> None:
    """Confirm ingest produced the per-chunk artifacts we need.

    Raises FileNotFoundError with a clear message if anything is missing, so
    the failure surfaces at orchestration time rather than deep inside deskew.
    """
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
    """Run A→B→C for a single chunk.  Returns (chunk_id, ok, error_msg).

    Per-chunk input validation lives inside the try/except so that a chunk
    with partial ingest artifacts surfaces a clear error message AND the
    other chunks keep running.

    On failure, error_msg includes the formatted traceback so it survives
    the worker -> parent boundary in ProcessPoolExecutor.
    """
    try:
        _validate_chunk_inputs(bag_id, chunk_id)

        log.info("=== chunk %s: step A — deskew ===", chunk_id)
        deskew.process_chunk(cfg, bag_id, chunk_id)

        # Step A.5 — MF-MOS (no-op when cfg.mf_mos.enabled is False).
        # Runs before classify so classify can optionally fuse both signals.
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

        # Roll per-sweep counts + runtime stats into one queryable row so
        # downstream stages can find pathological chunks (cache disabled,
        # zero ground returns, lots of invalid sweeps) without iterating
        # lidar_proc_index.
        _write_chunk_summary(bag_id, chunk_id, classify_result, ground_result, mf_mos_result)
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
) -> classify.ClassifyResult:
    """Module-level worker for ProcessPoolExecutor.

    Local closures are not picklable so cannot be sent to subprocesses; this
    function must stay at module scope.  Each worker builds its own
    GlobalMapPrior from the on-disk path to avoid pickling a large cKDTree
    across the process-pool pipe.
    """
    prior = classify.GlobalMapPrior.from_npz(
        global_map_path,
        match_radius_m=cfg.global_map_match_radius_m,
        r_max_credibility_m=cfg.r_max_credibility_m,
    )
    return classify.process_chunk(cfg, bag_id, chunk_id, global_map_prior=prior)


def _run_classify_pass2(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str | None,
    workers: int,
    global_map_path: str,
) -> None:
    """Re-run classify (only) on every chunk with the global map as a prior.

    Pass 2 deliberately skips deskew, mf_mos, and ground — their outputs are
    already on disk and don't depend on the global map.  It rewrites
    static_map.npz / dynamic_map.npz / lidar_proc_index for each chunk.
    """
    chunk_rows = read_rows(chunks_index_path(bag_id))
    if chunk_id:
        log.warning(
            "two-pass mode with --chunk %s: global map was built from a single chunk only. "
            "Run without --chunk to use the full bag map as prior.",
            chunk_id,
        )
        chunk_rows = [r for r in chunk_rows if r["chunk_id"] == chunk_id]

    n_total = len(chunk_rows)
    n_ok = 0
    failures: list[tuple[str, str]] = []

    if workers <= 1:
        # Single-worker path: build the KDTree once and reuse across all chunks
        # in this process.  Avoids rebuilding a million-point KDTree N times.
        prior = classify.GlobalMapPrior.from_npz(
            global_map_path,
            match_radius_m=cfg.global_map_match_radius_m,
            r_max_credibility_m=cfg.r_max_credibility_m,
        )
        for row in chunk_rows:
            cid = row["chunk_id"]
            try:
                classify.process_chunk(cfg, bag_id, cid, global_map_prior=prior)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001 — one chunk failing must not stop the rest
                log.exception("pass 2 chunk %s failed", cid)
                failures.append((cid, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    else:
        # Multi-worker path: each worker builds its own KDTree from disk path.
        # Pickling cKDTree across the pool pipe per chunk is slower than
        # building from the on-disk path inside each worker process.
        worker = functools.partial(
            _pass2_chunk_worker,
            cfg=cfg,
            bag_id=bag_id,
            global_map_path=global_map_path,
        )
        log.info("pass 2: running %d chunks across %d workers", n_total, workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(worker, r["chunk_id"]): r["chunk_id"] for r in chunk_rows}
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
        log.warning("pass 2 failed chunks:")
        for cid, err in failures:
            log.warning("--- chunk %s ---\n%s", cid, err)


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
        workers: number of concurrent worker processes (>=1).
        two_pass: defaults to True.  After pass 1 completes, builds the
            bag-level global_static_map.npz via reduce_static_map and re-runs
            classify (only) on every chunk with that map as a per-sweep KDTree
            prior (UniLiPs IWU).  Roughly doubles wall time but improves static
            recall on long-range structure sparsely observed in any one chunk.
            Set False for the legacy single-pass behavior.
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

    pending: list[str] = []
    skipped: list[str] = []
    for row in chunk_rows:
        cid = row["chunk_id"]
        if not force and _chunk_complete(bag_id, cid):
            skipped.append(cid)
            continue
        pending.append(cid)

    if skipped:
        log.info(
            "skipping %d already-processed chunks (use force=True to re-run): %s",
            len(skipped),
            skipped[:5] + (["..."] if len(skipped) > 5 else []),
        )

    if not pending:
        log.info(
            "lidar_preprocessing: all %d chunks already processed for bag %s",
            len(chunk_rows),
            bag_id,
        )
        return

    n_total = len(pending)
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

    log.info(
        "lidar_preprocessing complete for bag %s: %d/%d chunks succeeded",
        bag_id,
        n_ok,
        n_total,
    )
    if failures:
        log.warning("failed chunks:")
        for cid, err in failures:
            # err already includes a formatted traceback from worker side.
            log.warning("--- chunk %s ---\n%s", cid, err)

    if n_ok == 0 and n_total > 0:
        raise RuntimeError(
            f"lidar_preprocessing: all {n_total} chunks failed for bag {bag_id!r}"
        )

    if two_pass:
        log.info(
            "two-pass mode: building bag-level global_static_map.npz from pass-1 results"
        )
        global_map_uri = reduce_static_map(bag_id, cfg)
        global_map_path_str = local_path(global_map_uri)
        log.info(
            "two-pass mode: starting pass 2 (classify only, global map prior at %s)",
            global_map_path_str,
        )
        _run_classify_pass2(cfg, bag_id, chunk_id, workers, global_map_path_str)


__all__ = ["run", "_chunk_complete", "_process_one_chunk"]
