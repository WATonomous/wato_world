# LiDAR preprocessing contract repair handoff

Last updated: 2026-08-20 on branch `JD_lidar_preprocessing_cleanup`.

## Objective

Implement items 1–5 of the approved LiDAR Preprocessing Contract Repair Plan
without replacing Adaptive-Witch, MF-MOS, Patchwork++, global-prior, deskew, or
reduction algorithms. Native multi-LiDAR processing is authoritative:
`(lidar_id, sweep_id)` identifies a physical sweep; `frame_id` only groups
synchronized sweeps.

## Current implementation state

The working tree contains the implementation but no new commits. Git metadata
is read-only in the current sandbox and the escalation service was quota-blocked
when commit creation was attempted. Preserve all current changes.

Implemented:

- v2 LiDAR storage rooted at
  `lidar_preprocessing/v2/<bag>/<chunk>/sweeps/<lidar_id>/<sweep_id>_*`, with
  v2 downstream pins and safe case-sensitive LiDAR IDs.
- Composite keys across pose validity, deskew, MF-MOS metadata, processed
  indexes, perception caches/loaders, semantic lifting, visualization lookup,
  and dynamic-map provenance.
- `LidarProfile` / `MFMosProfile` resolution with strict unknown-sensor behavior,
  heterogeneous timing/geometry/height, and shared-AW contract validation.
- Explicit point-aligned, disjoint static and dynamic masks. Unknown points are
  in neither. Final static is `AW_STATIC & ~dynamic & ~ground_candidate`.
- MF-MOS per-sensor history/geometry and persisted statuses. Empty/all-false
  success is `ok`; strict `mfmos_only` fails on every unavailable valid sweep;
  union degrades to AW with a warning.
- Final ground is exactly `ground_candidate & ~dynamic_mask`; static voxel
  membership is no longer required. Candidate/final counts and
  `n_rejected_dynamic_ground` are persisted.
- Transactional staging for one-pass and two-pass runs. Full-bag pass one is
  staged across every chunk, the prior is built from staged maps, pass two
  reruns classify/ground/summary, artifacts validate, then promotion occurs.
  Chunk-local two-pass uses a staged local prior and does not publish bag
  globals.
- `completion.json` and bag `manifest.json` contain version, mode, config
  digest, run ID, and composite counts. Skip requires an exact completion
  contract match; `ground.npz` alone is insufficient.
- Perception depth anchors load `static_mask_path` directly; semantic lifting
  and per-sweep visualization carry `lidar_id`. Ambiguous visualization sweep
  IDs require `--lidar-id`.
- Root/component/ingest/pipeline/perception/semantic-lifting/SAM4D/SLF docs now
  explain native versus merged input, identity versus grouping, profiles,
  masks/ground, strict MF-MOS, publication modes, v2 layout, and reprocessing.

## Verification evidence

Focused tests run successfully before the latest final audit edits:

- LiDAR profiles: 8 passed
- deskew: 27 passed
- classify: 23 passed
- MF-MOS: 27 passed
- ground: 11 passed
- ingest artifact store: 14 passed after publication path/staging tests
- ingest frame index: 5 passed
- perception LiDAR I/O regression: 1 passed
- semantic-lifting LiDAR I/O regression: 1 passed
- pipeline orchestration: 11 passed before adding the final three-sensor and
  missing-mask validation regressions
- visualization composite lookup: 3 passed

Latest non-container checks:

- `python3 -m compileall -q` over common, ingest, LiDAR preprocessing,
  perception, and semantic lifting: passed.
- `git diff --check`: passed.
- Repository-wide searches found no remaining production call using the old
  three-argument per-sweep path/load APIs.

Container verification became unavailable because Docker requires escalation
and the approval service reported its usage quota exhausted until 1:03 PM.
Host Python lacks pytest and Pydantic, so do not claim the final suites pass yet.

## Exact next steps

1. Run the complete LiDAR suite in the existing dependency image:

   ```bash
   docker run --rm --entrypoint python3 \
     -v "$PWD/src/common:/ws/src/common:ro" \
     -v "$PWD/src/lidar_preprocessing:/ws/src/lidar_preprocessing:ro" \
     -v "$PWD/config:/config:ro" \
     -e PYTHONPATH=/ws/src/common/src:/ws/src/lidar_preprocessing/src \
     ghcr.io/watonomous/wato_world/lidar_preprocessing:dev_JD_lidar_preprocessing_cleanup \
     -m pytest /ws/src/lidar_preprocessing/tests -q -p no:cacheprovider
   ```

   Pay particular attention to the newly added pipeline tests:
   `test_failed_staged_rerun_preserves_existing_publication`,
   `test_three_sensor_duplicate_sweep_ids_survive_canonical_grouping`, and
   `test_validation_rejects_missing_point_aligned_mask`.

2. Run ingest/common-schema, perception, and semantic-lifting suites in their
   dependency-equipped containers. Fix failures test-first; use systematic
   debugging before changing behavior.

3. Run formatting/lint tooling available in the component images. Re-run
   `python3 -m compileall -q ...` and `git diff --check`.

4. Review transactional publication once more for failure during promotion
   itself. Pre-promotion failures and restoration of an existing publication
   are covered; consider adding a mocked `os.replace` failure test for rollback
   of chunks, reductions, and the prior manifest.

5. Create reviewable commits once `.git` writes are permitted. Recommended
   grouping:

   - `feat(lidar): repair v2 multi-sensor processing contract` — production
     code, schemas/config pins, and tests.
   - `docs(lidar): document v2 multi-sensor publication contract` — READMEs
     and research guidance.

6. Inspect `git status --short` and commit only the files in this handoff's
   scope. Do not reset or discard the dirty working tree.

## Known review notes

- `WATO_LIDAR_STAGING_ROOT` redirects only local
  `/lidar_preprocessing/v2/` paths. Raw ingest reads remain canonical, and
  worker processes inherit the redirect via the environment.
- Full-bag runs reprocess all bag chunks if any selected completion record is
  stale, ensuring one prior/reduction lineage. Chunk-only runs may skip their
  one exact record independently.
- MF-MOS masks remain raw-NPZ aligned; classify converts them to the filtered
  world-sweep alignment before fusion. Final static/dynamic masks are directly
  world-sweep aligned.
- Standalone `reduce` remains available, but transactional full-bag pipeline
  runs publish reductions themselves. The CLI `--auto-reduce` flag is retained
  only as deprecated compatibility syntax.
- v1 artifacts are deliberately untouched and unread. Reprocessing is
  mandatory.
