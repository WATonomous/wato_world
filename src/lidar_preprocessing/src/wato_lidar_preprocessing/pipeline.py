"""lidar_preprocessing pipeline orchestrator.

Runs Steps A → B → C for each chunk. Step B is one of three segmentation
methods, selected by cfg.segmentation (`--seg aw|mos|union`):

  A.  deskew        — motion compensation + world-frame projection (+ Patchwork++)
  B.  static/dynamic decomposition:
        seg=aw    → classify (Amanatides-Woo log-odds ray-casting). No MF-MOS.
        seg=mos   → mf_mos inference + mf_mos segmentation. No ray traversal.
        seg=union → classify (static basis) + mf_mos inference + union fusion:
                    keep AW's static map, take MF-MOS dynamics vetoed by it.
  C.  ground        — per-sweep ground aggregation + height grid

The two base methods (aw, mos) never import each other; each writes the same
artifacts (static_map / dynamic_map / dynamic_mask / index) so C and D are
method-agnostic. `union` is the fusion layer — it runs both halves and reads
their outputs, then rewrites only the dynamic side. On the union path Step C
runs BEFORE the fusion (union's ground-height veto reads ground.npz's height
grid; ground itself only needs static_map.npz, so the swap is safe).

Step D (reduce) runs separately via the `reduce` CLI subcommand. In
two_pass mode (aw/union), run() invokes reduce + a classify-only pass 2.

Steps E (iwu, bag-level UniLiPs IWU) and F (motion_proposals, per-chunk
recall-oriented moving-object proposals) run after reduce via
run_proposals(). Both are seg-agnostic consumers of Steps B–D.

Idempotency: chunks whose summary parquet exists (written last, after every
step) for the same segmentation method are skipped unless force=True.
Failures in one chunk don't stop the others.
"""

from __future__ import annotations

import functools
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunks_index_path,
    dynamic_map_path,
    frame_index_path,
    global_iwu_path,
    ground_path,
    lidar_proc_index_path,
    lidar_proc_summary_path,
    lidar_sweeps_path,
    local_path,
    motion_clusters_path,
    poses_path,
    static_map_path,
)
from wato_common.io.parquet_io import read_rows, write_table
from wato_common.schemas import CHUNK_SUMMARY_SCHEMA, ChunkSummaryRow
from wato_lidar_preprocessing import (
    classify,
    deskew,
    ground,
    iwu as iwu_step,
    mf_mos as mf_mos_step,
    motion_proposals,
    union as union_step,
)
from wato_lidar_preprocessing.config import ComponentConfig
from wato_lidar_preprocessing.reduce import reduce_static_map

log = logging.getLogger(__name__)


def _write_chunk_summary(
    bag_id: str,
    chunk_id: str,
    segmentation: str,
    seg_result: (
        "classify.ClassifyResult | mf_mos_step.MosSegmentResult "
        "| union_step.UnionSegmentResult"
    ),
    ground_result: ground.GroundResult,
    mf_mos_result: mf_mos_step.MFMosResult | None = None,
) -> None:
    """Aggregate per-sweep parquet + segmentation/ground results into one row.

    Per-sweep counts come from lidar_proc_index (after Step B updated it).
    Runtime stats come from the step result objects. `seg_result` is the AW
    ClassifyResult, the MF-MOS MosSegmentResult, or the UnionSegmentResult;
    all three expose the cache fields the summary reads, and the mos/union
    extras (n_sweeps_no_mask, n_vetoed) are read with getattr so the others
    record None.
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
        cache_auto_disabled=seg_result.cache_auto_disabled,
        estimated_cache_bytes=seg_result.estimated_cache_bytes,
        ground_status=ground_result.status,
        segmentation_method=segmentation,
        seg_n_sweeps_no_mask=getattr(seg_result, "n_sweeps_no_mask", None),
        union_n_points_vetoed=getattr(seg_result, "n_vetoed", None),
        union_n_points_ground_vetoed=getattr(seg_result, "n_ground_vetoed", None),
        motion_filter_n_persistence_dropped=getattr(
            seg_result, "n_persistence_dropped", None
        ),
        motion_filter_n_coherence_dropped=getattr(
            seg_result, "n_coherence_dropped", None
        ),
        mf_mos_n_processed=mf_mos_result.n_sweeps_processed if mf_mos_result else None,
        mf_mos_n_skipped=mf_mos_result.n_skipped if mf_mos_result else None,
        mf_mos_n_unsupported=(
            mf_mos_result.n_sweeps_skipped_unsupported if mf_mos_result else None
        ),
        mf_mos_n_points_moving=mf_mos_result.n_points_moving if mf_mos_result else None,
    )
    write_table(
        [summary.model_dump()],
        CHUNK_SUMMARY_SCHEMA,
        lidar_proc_summary_path(bag_id, chunk_id),
    )


def _chunk_complete(bag_id: str, chunk_id: str, segmentation: str) -> bool:
    """Chunk done for THIS run's segmentation method; re-running won't help.

    The chunk summary is the completeness marker — it is written last in
    _process_one_chunk, after every step, so its presence means the chunk
    finished. ground.npz alone is NOT enough: on the union path Step C runs
    before the fusion, so ground.npz can exist for a chunk whose dynamic
    side was never written.

    Artifacts are also only reusable if the same Step-B method produced them
    — comparing `--seg` runs against each other must not silently serve
    stale outputs. Summaries written before segmentation_method existed
    can't be verified and are trusted (legacy behavior).
    """
    if not os.path.exists(local_path(ground_path(bag_id, chunk_id))):
        return False
    summary_uri = lidar_proc_summary_path(bag_id, chunk_id)
    if not os.path.exists(local_path(summary_uri)):
        return False  # crashed mid-chunk (or pre-summary artifacts): redo
    rows = read_rows(summary_uri)
    recorded = rows[0].get("segmentation_method") if rows else None
    if recorded is None:
        return True  # legacy summary without the column
    if recorded != segmentation:
        log.info(
            "chunk %s: artifacts were produced by seg=%s but this run is seg=%s "
            "— re-processing",
            chunk_id,
            recorded,
            segmentation,
        )
        return False
    return True


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


def _write_chunk_manifest(
    bag_id: str,
    chunk_id: str,
    config_path: str | None,
    *,
    with_proposals: bool = False,
) -> None:
    """Record what this chunk was produced from, and by what.

    Best-effort: a manifest failure must never fail a chunk that otherwise
    succeeded — losing traceability for one chunk is better than discarding
    the compute that produced it. Step F rewrites it with with_proposals=True
    so the manifest also covers motion_clusters.parquet and the bag-level
    global_iwu.npz it read (hashed like every other input).
    """
    inputs = {
        "lidar_sweeps": lidar_sweeps_path(bag_id, chunk_id),
        "poses": poses_path(bag_id, chunk_id),
        "frame_index": frame_index_path(bag_id, chunk_id),
    }
    outputs = {
        "static_map": static_map_path(bag_id, chunk_id),
        "dynamic_map": dynamic_map_path(bag_id, chunk_id),
        "ground": ground_path(bag_id, chunk_id),
        "lidar_proc_index": lidar_proc_index_path(bag_id, chunk_id),
        "lidar_proc_summary": lidar_proc_summary_path(bag_id, chunk_id),
    }
    if with_proposals:
        if os.path.exists(local_path(global_iwu_path(bag_id))):
            inputs["global_iwu"] = global_iwu_path(bag_id)
        outputs["motion_clusters"] = motion_clusters_path(bag_id, chunk_id)
    try:
        common_manifest.write(
            component="lidar_preprocessing",
            bag_id=bag_id,
            chunk_id=chunk_id,
            inputs=inputs,
            outputs=outputs,
            config_path=config_path,
            filename=common_manifest.component_manifest_name("lidar_preprocessing"),
        )
    except Exception:  # noqa: BLE001
        log.warning("failed to write manifest for chunk %s", chunk_id, exc_info=True)


def _process_one_chunk(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    config_path: str | None = None,
) -> tuple[str, bool, str]:
    """Run A → A.5 → B → C for a single chunk.

    Returns (chunk_id, ok, error_msg). error_msg carries the full traceback
    so it survives the ProcessPoolExecutor worker boundary.

    On success, writes manifest_lidar_preprocessing.json recording the ingest
    artifacts consumed (hashed, so a later re-ingest is detectable) and the
    image provenance that produced the outputs.
    """
    try:
        _validate_chunk_inputs(bag_id, chunk_id)

        log.info("=== chunk %s: step A — deskew ===", chunk_id)
        deskew.process_chunk(cfg, bag_id, chunk_id)

        # Step B: one of two fully independent segmentation methods. They do
        # not coexist — aw never runs MF-MOS, mos never runs ray traversal.
        if cfg.segmentation == "mos":
            log.info("=== chunk %s: step B — mf_mos inference ===", chunk_id)
            mf_mos_result = mf_mos_step.process_chunk(cfg, bag_id, chunk_id)
            log.info("=== chunk %s: step B — mf_mos segmentation ===", chunk_id)
            seg_result = mf_mos_step.classify_chunk(cfg, bag_id, chunk_id)
            log.info("=== chunk %s: step C — ground ===", chunk_id)
            ground_result = ground.process_chunk(cfg, bag_id, chunk_id)
        elif cfg.segmentation == "union":
            # Fusion: AW builds the static basis, MF-MOS proposes motion, the
            # union step keeps AW's static map and vetoes MF-MOS dynamics that
            # land on it. Step C runs BEFORE the fusion here — union's
            # ground-height veto reads ground.npz's height grid (ground only
            # needs static_map.npz, so the swap is safe).
            log.info(
                "=== chunk %s: step B — classify (Amanatides-Woo static basis) ===",
                chunk_id,
            )
            aw_result = classify.process_chunk(cfg, bag_id, chunk_id)
            log.info("=== chunk %s: step B — mf_mos inference ===", chunk_id)
            mf_mos_result = mf_mos_step.process_chunk(cfg, bag_id, chunk_id)
            log.info("=== chunk %s: step C — ground (before fusion) ===", chunk_id)
            ground_result = ground.process_chunk(cfg, bag_id, chunk_id)
            log.info("=== chunk %s: step B — union fusion ===", chunk_id)
            seg_result = union_step.classify_chunk(
                cfg, bag_id, chunk_id, aw_result=aw_result
            )
        else:  # "aw"
            log.info("=== chunk %s: step B — classify (Amanatides-Woo) ===", chunk_id)
            mf_mos_result = None
            seg_result = classify.process_chunk(cfg, bag_id, chunk_id)
            log.info("=== chunk %s: step C — ground ===", chunk_id)
            ground_result = ground.process_chunk(cfg, bag_id, chunk_id)

        # A disabled step reports None, not zeros — "didn't run" and "ran and
        # found nothing" must stay distinguishable in the summary.
        _write_chunk_summary(
            bag_id, chunk_id, cfg.segmentation, seg_result, ground_result, mf_mos_result
        )
        _write_chunk_manifest(bag_id, chunk_id, config_path)
        return (chunk_id, True, "")
    except Exception as exc:  # noqa: BLE001 — one chunk failing must not stop the rest
        log.exception("chunk %s failed", chunk_id)
        tb = traceback.format_exc()
        return (chunk_id, False, f"{type(exc).__name__}: {exc}\n{tb}")


def _classify_then_maybe_fuse(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str,
    prior: "classify.GlobalMapPrior",
) -> "classify.ClassifyResult | union_step.UnionSegmentResult":
    """Pass-2 Step B for one chunk.

    Re-classify with the global-map prior. For `union`, re-fuse afterwards so
    the improved (prior-boosted) AW static map re-vetoes the MF-MOS dynamics.
    MF-MOS masks are unchanged from pass 1 (the prior is AW-only), so inference
    is not re-run — union just re-reads them.
    """
    res = classify.process_chunk(cfg, bag_id, chunk_id, global_map_prior=prior)
    if cfg.segmentation == "union":
        return union_step.classify_chunk(cfg, bag_id, chunk_id, aw_result=res)
    return res


def _pass2_chunk_worker(
    chunk_id: str,
    cfg: ComponentConfig,
    bag_id: str,
    global_map_path: str,
) -> "classify.ClassifyResult | union_step.UnionSegmentResult":
    """ProcessPoolExecutor worker. Must stay module-scope (closures aren't
    picklable). Each worker rebuilds the KDTree from disk rather than pickling
    a large cKDTree across the pool pipe.
    """
    # Match radius = the map's own voxel size: reduce snaps map points to
    # voxel centres, so "within one map voxel" is what a match means.
    prior = classify.GlobalMapPrior.from_npz(
        global_map_path,
        match_radius_m=cfg.global_map_voxel_size_m,
    )
    return _classify_then_maybe_fuse(cfg, bag_id, chunk_id, prior)


def _run_classify_pass2(
    cfg: ComponentConfig,
    bag_id: str,
    chunk_id: str | None,
    workers: int,
    global_map_path: str,
) -> None:
    """Re-run classify on every chunk with the global map as a prior.

    Skips deskew/ground — their outputs are unchanged. For `union`, also
    re-runs the fusion (but not MF-MOS inference, whose masks are unchanged
    since the prior is AW-only). Rewrites static_map.npz / dynamic_map.npz /
    lidar_proc_index per chunk.
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
        # Build the KDTree once and reuse across chunks in this process.
        prior = classify.GlobalMapPrior.from_npz(
            global_map_path,
            match_radius_m=cfg.global_map_voxel_size_m,
        )
        for row in chunk_rows:
            cid = row["chunk_id"]
            try:
                _classify_then_maybe_fuse(cfg, bag_id, cid, prior)
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
    two_pass: bool = False,
    config_path: str | None = None,
) -> None:
    """Process all chunks (or one) for a bag.

    Args:
        cfg: parsed ComponentConfig.
        bag_id: bag identifier (must have been ingested).
        chunk_id: optional single-chunk filter.
        force: re-process chunks whose ground.npz already exists.
        workers: concurrent worker processes (>=1).
        two_pass: when True (default False), after pass 1 builds the
            bag-level global_static_map.npz via reduce_static_map and re-runs
            classify on every chunk using that map as a global-map prior (a
            one-time log-odds boost for map-matched voxels — not UniLiPs IWU,
            which is the bag-level `iwu` step). Roughly doubles classify wall
            time; improves static recall on long-range structure sparsely
            observed in any one chunk.
        config_path: path to the config file that produced ``cfg``. Recorded
            (as a hash) in each chunk's manifest so a label can be traced back
            to the exact parameters that produced it. Optional only so the
            existing tests can call run() without one.
    """
    # The global-map prior is an Amanatides-Woo log-odds boost.
    # `mos` has no log-odds to boost, so it runs single-pass. `union` does have
    # an AW half, and a better static map sharpens the dynamic veto, so it
    # keeps two-pass (pass 2 re-classifies + re-fuses; MF-MOS is not re-run).
    if two_pass and cfg.segmentation == "mos":
        log.info("seg=mos: two-pass global-map prior is aw-only — running single-pass")
        two_pass = False

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
        if not force and _chunk_complete(bag_id, cid, cfg.segmentation):
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
            _, ok, err = _process_one_chunk(cfg, bag_id, cid, config_path)
            if ok:
                n_ok += 1
            else:
                failures.append((cid, err))
    else:
        log.info("running %d chunks across %d workers", n_total, workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_one_chunk, cfg, bag_id, cid, config_path): cid
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


def _proposals_chunk_worker(
    chunk_id: str, cfg: ComponentConfig, bag_id: str, config_path: str | None
) -> tuple[str, bool, str]:
    """Step F for one chunk. Module-scope so ProcessPoolExecutor can pickle it."""
    try:
        motion_proposals.process_chunk(cfg, bag_id, chunk_id)
        _write_chunk_manifest(bag_id, chunk_id, config_path, with_proposals=True)
        return (chunk_id, True, "")
    except Exception as exc:  # noqa: BLE001 — one chunk failing must not stop the rest
        log.exception("proposals chunk %s failed", chunk_id)
        return (
            chunk_id,
            False,
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def run_proposals(
    cfg: ComponentConfig,
    *,
    bag_id: str,
    chunk_id: str | None = None,
    force: bool = False,
    workers: int = 1,
    config_path: str | None = None,
    with_iwu: bool = True,
) -> None:
    """Steps E + F: bag-level IWU, then per-chunk motion proposals.

    Needs Steps A–D done (global_static_map.npz for IWU; each chunk's
    lidar_proc_index / static_map / dynamic masks for F).

    Args:
        chunk_id: restrict F to one chunk. IWU is bag-level, so it is skipped
            and F uses whatever global_iwu.npz already exists (IWU_EVICTED is
            never set when none does).
        force: re-run F on chunks whose motion_clusters.parquet is newer than
            every input it read.
        with_iwu: False skips Step E (the `proposals` subcommand; `iwu` is its
            own subcommand for multi-machine runs).
    """
    if with_iwu and chunk_id is None and cfg.iwu.enabled:
        log.info("=== bag %s: step E — iwu ===", bag_id)
        try:
            iwu_step.run_iwu(cfg, bag_id)
        except FileNotFoundError as exc:
            log.warning("bag %s: IWU skipped (%s)", bag_id, exc)

    if not cfg.motion_proposals.enabled:
        log.info("motion_proposals.enabled=false — skipping step F")
        return

    chunk_rows = read_rows(chunks_index_path(bag_id))
    if chunk_id:
        chunk_rows = [r for r in chunk_rows if r["chunk_id"] == chunk_id]
    pending: list[str] = []
    for r in chunk_rows:
        cid = r["chunk_id"]
        if not os.path.exists(local_path(lidar_proc_summary_path(bag_id, cid))):
            log.warning("chunk %s: not processed by steps A–C — no proposals", cid)
            continue
        if not force and motion_proposals.proposals_up_to_date(bag_id, cid):
            continue
        pending.append(cid)
    log.info(
        "=== bag %s: step F — motion proposals on %d chunk(s) (%d up to date) ===",
        bag_id,
        len(pending),
        len(chunk_rows) - len(pending),
    )
    if not pending:
        return

    worker = functools.partial(
        _proposals_chunk_worker, cfg=cfg, bag_id=bag_id, config_path=config_path
    )
    failures: list[tuple[str, str]] = []
    if workers <= 1:
        for cid in pending:
            _, ok, err = worker(cid)
            if not ok:
                failures.append((cid, err))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for cid, ok, err in pool.map(worker, pending):
                if not ok:
                    failures.append((cid, err))
    if failures:
        log.warning("step F failed chunks:")
        for cid, err in failures:
            log.warning("--- chunk %s ---\n%s", cid, err)


__all__ = ["run", "run_proposals", "_chunk_complete", "_process_one_chunk"]
