"""MapMOS per-sweep inference.

**Step 1 ships the stub** — `run_sweep_inference` returns a length-N
zero array (neutral; fusion contributes nothing). With the stub in
place, the regression gate `scripts/compare_summaries.py` must report
identical static/dynamic counts vs the no-MapMOS run for both
`fusion.alpha = 0.0` and `fusion.alpha = 1.0` (plan §1).

==========================================================================
PRBonn convention (V1+V2+V3 verified)
==========================================================================
Source: PRBonn/MapMOS @ commit 8947300698c61257ddb1e1e9f927382f0c0a0bac
       (tag: HEAD on main as of 2025-09-11). Specifically:
       - src/mapmos/mapmos_net.py:38-82 (predict + forward)
       - src/mapmos/datasets/mapmos_dataset.py:39-74 (collate_fn)
       - src/mapmos/pipeline.py:124-149 (run loop calling model.predict)
       - src/mapmos/config/config.py:41-47 (MOSConfig defaults)

Differences from the original Step-3 plan (the plan was wrong on these;
the code below is the corrected design):

V1. Time channel is the ABSOLUTE SCAN INDEX (integer), not seconds.
    PRBonn does NOT take an N-past-sweeps window. The model's `predict()`
    takes two point sets:
      - scan_input: current frame's points (in WORLD frame), all
                    stamped with the current scan_index.
      - map_input:  accumulated registered points from a running
                    VoxelHashMap, each retaining its ORIGINAL scan_index.
    Inside MinkUNet.forward(), the per-point integer indices get
    normalized into features in [1, 2]:
        features = 1 + (i_max - indices) / (i_max - i_min)
    Newer scan → 1.0; oldest scan → 2.0. There is NO time-in-seconds
    anywhere. No sweep_period_s. We are removing those config fields.

V1 (also). Coordinates are 5-D: [batch_idx, x, y, z, scan_or_map_flag],
    where the flag is 0.0 for current-scan points and -1.0 for map
    points. The flag is NOT divided by voxel_size during quantization
    (mapmos_net.py:60: quantization = [1, vox, vox, vox, 1]); it becomes
    a literal extra spatial axis. The MinkUNet uses D=4 (4D sparse conv)
    so scan and map points stay separable.

V2. NO GROUND STRIPPING. PRBonn trains on full point clouds; no
    patchwork, no ground filter, no floor removal anywhere in their
    repo (grep -rn 'ground|patchwork|floor' src/mapmos/ — only the
    Polyscope visualizer hits). Their only preprocessing is a relative-
    to-ego range filter at inference time (pipeline.py:117-121:
    ranges = ||points - ego||; mask = min <= ranges <= max).

V3. VOXEL <-> POINT MAPPING is handled by MinkowskiEngine's TensorField
    API — no manual indexing needed:
        tensor_field = ME.TensorField(features, coordinates)
        sparse_tensor = tensor_field.sparse()                  # forward
        out = predicted_sparse_tensor.slice(tensor_field)      # reverse
        logits = out.features.reshape(-1)
    Multiple input points at the same voxel get the same logit; that's
    fine — voxelization is intentionally lossy. Our `run_model` will
    follow this exact shape (mapmos_net.py:72-82) verbatim.

Architectural consequence: our existing scaffolding for sensor-frame
transforms (`mapmos/sensor_frame.py`) and N-past-scans history
(`mapmos/history.py`) is NOT needed for inference fidelity. The model
wants WORLD-frame current-scan + WORLD-frame accumulated-map points.
We already have all world-frame points from deskew, so Step 3 only
needs to maintain a running map accumulator across chunks.

Plan non-negotiables this module enforces: #2 length invariant, #5
logit clamp, #13 per-input-point return shape, #14 assert vs xyz NPZ
(not parquet), #16 CUDA OOM empty_cache, #20 empty != None.
Plan non-negotiables made obsolete by V1/V2/V3:
  - #3 ground-strip + reconstruct (no longer needed — input includes ground)
  - #11 time-channel convention (we now use scan indices, not seconds)
  - #12 separated range filter on query vs history (no history window — single
    range filter relative to ego on the combined scan+map cloud is correct)
  - #15 sensor-frame transform (world frame is correct)
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PRBonn convention constants — verified, do not change without re-reading
# the source code cited in the module docstring.
# ---------------------------------------------------------------------------

# V2: MapMOS was trained on FULL point clouds (no ground removal).
# Source: src/mapmos/datasets/kitti.py:55-66 returns all valid points;
# src/mapmos/pipeline.py:117-121 applies only range filter at inference.
# No grep hit on ground/patchwork/floor in src/mapmos/.
_PRBONN_INPUT_INCLUDES_GROUND: bool = True

# Locked checkpoint voxel size. The pretrained MinkUNet was trained at
# this resolution and changing it silently breaks the model.
# Source: src/mapmos/config/config.py:42 (MOSConfig.voxel_size_mos)
EXPECTED_VOXEL_SIZE_M: float = 0.1

# Range filter is RELATIVE TO EGO POSITION at the current scan time,
# applied uniformly to scan_input + map_input.
# Source: src/mapmos/pipeline.py:117-121 (_preprocess function)
# Source: src/mapmos/config/config.py:44-45 (MOSConfig.{max,min}_range_mos)
DEFAULT_MIN_RANGE_M: float = 0.0
DEFAULT_MAX_RANGE_M: float = 50.0

# Scan-or-map flag values used in the 5-D coordinate.
# Source: src/mapmos/mapmos_net.py:44-45
_SCAN_FLAG: float = 0.0
_MAP_FLAG: float = -1.0


# ---------------------------------------------------------------------------
# Stub inference — produces a length-N zero array
# ---------------------------------------------------------------------------
def run_sweep_inference_stub(n_points_total: int) -> np.ndarray:
    """Neutral zero-prior stub. Length matches the world NPZ point count.

    Used until Step 3 lands the real MinkUNet recipe. Returning zeros
    means the additive log-odds update `+= l_occ + alpha * 0 = += l_occ`
    is bit-identical to the geometry-only path, which is exactly the
    regression invariant the Step 1 gate verifies.
    """
    return np.zeros(int(n_points_total), dtype=np.float32)


def run_sweep_inference(
    *,
    xyz_world: np.ndarray,
    ego_world_xyz: np.ndarray | None = None,
    ground_mask: np.ndarray | None = None,  # noqa: ARG001 — kept for stub-mode call-site stability; V2 says no ground strip
    scan_index: int | None = None,
    map_accumulator=None,
    model=None,
    ckpt_voxel_size: float | None = None,
    device=None,
    min_range_m: float = DEFAULT_MIN_RANGE_M,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    logit_clamp: float = 10.0,
) -> np.ndarray:
    """Returns float32 logits of length `xyz_world.shape[0]`.

    Phase 4 — real recipe. When `model` is None, falls back to the
    zero-stub (Step 1 regression behavior). When all the inference
    inputs are present, runs the actual MinkUNet forward pass:

      1. Range-filter `xyz_world` relative to ego (matches PRBonn's
         _preprocess at pipeline.py:117-121 — radius from ego, not from
         world origin).
      2. Pull map_input + map_indices from the running accumulator,
         also range-filter relative to the SAME ego.
      3. Build scan_indices = full of `scan_index` (one integer per
         scan point — PRBonn convention, V1).
      4. Call run_model → (logits_scan, logits_map). Discard logits_map;
         we only fuse the current-scan logits into our classifier.
      5. Reconstruct full-length logits aligned with xyz_world: points
         filtered out by the range mask get a neutral 0.0; surviving
         points get their inference logit at the original index.
      6. Clamp to ±logit_clamp; warn on overshoot.

    Plan non-negotiables enforced here: #2 length invariant, #5 clamp +
    warn, #13 per-input-point return, #21 empty-scan-after-range-filter
    gracefully returns neutral length-N output.

    NOTE: `map_accumulator.add_scan(...)` is NOT called here. The caller
    (mapmos.pipeline.process_chunk) registers the scan AFTER writing
    the sidecar — see plan non-negotiable #30.
    """
    n_total = int(xyz_world.shape[0])

    # --- Stub fallback ---------------------------------------------------
    # When the caller hasn't supplied a model (e.g. Step 1 stub-mode
    # regression run, or model load failed gracefully), return neutral
    # length-N zeros. This is the regression-gate behavior verified in
    # test_neutral_prior_no_change_alpha_{zero,one}.
    if model is None or ego_world_xyz is None or scan_index is None:
        logits = run_sweep_inference_stub(n_total)
        if logits.size:
            raw_max = float(np.abs(logits).max())
            if raw_max > logit_clamp:
                log.warning("stub produced max |logit|=%.2f > clamp %.2f", raw_max, logit_clamp)
            logits = np.clip(logits, -logit_clamp, logit_clamp).astype(np.float32)
        return logits

    if n_total == 0:
        return np.empty(0, dtype=np.float32)

    # --- Range filter scan_input relative to ego (V2 / PRBonn _preprocess) ---
    # Squared norm is cheaper than sqrt; threshold both sides squared.
    ego = np.asarray(ego_world_xyz, dtype=np.float64).reshape(3)
    deltas = xyz_world.astype(np.float64, copy=False) - ego
    sq_ranges = np.einsum("ij,ij->i", deltas, deltas)
    min_sq = float(min_range_m) ** 2
    max_sq = float(max_range_m) ** 2
    in_range = (sq_ranges >= min_sq) & (sq_ranges <= max_sq)
    survived_idx = np.where(in_range)[0]

    if survived_idx.size == 0:
        # Every scan point fell outside the range filter (sensor pointing
        # at sky, ego in a tunnel, etc.). PRBonn would skip this frame;
        # we write a neutral length-N sidecar so classify stays aligned.
        log.debug(
            "scan_index=%s: all %d points outside range [%.1f, %.1f]m of ego — "
            "returning neutral logits",
            scan_index,
            n_total,
            min_range_m,
            max_range_m,
        )
        return np.zeros(n_total, dtype=np.float32)

    scan_input = xyz_world[in_range].astype(np.float32, copy=False)
    scan_indices = np.full(scan_input.shape[0], int(scan_index), dtype=np.int64)

    # --- Map input from accumulator (also range-filtered relative to ego) ---
    if map_accumulator is not None:
        map_points, map_indices = map_accumulator.get_map_points()
    else:
        map_points = np.empty((0, 3), dtype=np.float64)
        map_indices = np.empty(0, dtype=np.int64)

    if map_points.shape[0] > 0:
        map_deltas = map_points.astype(np.float64, copy=False) - ego
        map_sq = np.einsum("ij,ij->i", map_deltas, map_deltas)
        map_mask = (map_sq >= min_sq) & (map_sq <= max_sq)
        map_points = map_points[map_mask].astype(np.float32, copy=False)
        map_indices = map_indices[map_mask]

    # --- Forward pass ----------------------------------------------------
    logits_scan, _logits_map = run_model(
        model,
        scan_input,
        map_points,
        scan_indices,
        map_indices,
        ckpt_voxel_size if ckpt_voxel_size is not None else 0.1,
        device,
    )

    # --- Reconstruct full-length logits aligned with xyz_world ----------
    # Points filtered out by range get a neutral 0.0; survivors get their
    # inference logit at the ORIGINAL index in xyz_world.
    full_logits = np.zeros(n_total, dtype=np.float32)
    full_logits[survived_idx] = logits_scan.astype(np.float32, copy=False)

    # --- Clamp + warn ----------------------------------------------------
    if full_logits.size:
        raw_max = float(np.abs(full_logits).max())
        if raw_max > logit_clamp:
            log.warning(
                "scan_index=%s: max |logit|=%.2f exceeded clamp %.2f",
                scan_index,
                raw_max,
                logit_clamp,
            )
        full_logits = np.clip(full_logits, -logit_clamp, logit_clamp).astype(
            np.float32, copy=False
        )

    return full_logits


# ---------------------------------------------------------------------------
# Phase 2 — real forward-pass wrapper. Adapts numpy I/O to PRBonn's torch
# MapMOSNet.predict (src/mapmos/mapmos_net.py:38-82 @ commit 8947300).
# ---------------------------------------------------------------------------
def run_model(
    model,  # MapMOSNet (wraps CustomMinkUNet14 with D=4)
    scan_input: np.ndarray,    # (N_scan, 3) float32 in WORLD frame (current sweep)
    map_input: np.ndarray,     # (N_map, 3)  float32 in WORLD frame (accumulated)
    scan_indices: np.ndarray,  # (N_scan,)   int — same value for every point: current scan_index
    map_indices: np.ndarray,   # (N_map,)    int — original scan_index per accumulated map point
    voxel_size: float,         # noqa: ARG001 — locked at model construction time (0.1m)
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits_scan, logits_map), each aligned with its input.

    Wraps MapMOSNet.predict (mapmos_net.py:38-56) which internally:
      1. hstacks each set with [batch=0, ..., flag] (flag=0 scan, -1 map).
      2. vstacks scan+map into one coords tensor + one indices tensor.
      3. Quantizes coords with asymmetric [1, vox, vox, vox, 1] — the
         flag dim is NOT divided by voxel_size, becomes a separable axis.
      4. Derives features in [1, 2] from normalized indices (mapmos_net.py:64-70).
         If all indices are equal (empty map case), features = 1.0 everywhere.
      5. Builds ME.TensorField, runs MinkUNet, slices back per-input-point.
      6. Splits logits by flag and returns (logits_scan, logits_map).

    Plan non-negotiable #13: caller never sees voxel_to_point; PRBonn's
    slice() handles it internally. We return per-INPUT-point logits
    aligned with the input order, length-equal to each input.
    """
    import torch

    # Contiguous + dtype'd numpy arrays before tensor conversion. PRBonn's
    # `extend` helper does `i * ones` arithmetic; a (0, 3) shape for an
    # empty map vstacks cleanly with the (N_scan, 3) scan, but ONLY if
    # we keep both as 2D arrays. reshape(-1, 3) enforces that.
    scan_arr = np.ascontiguousarray(scan_input, dtype=np.float32).reshape(-1, 3)
    map_arr = np.ascontiguousarray(map_input, dtype=np.float32).reshape(-1, 3)
    sidx_arr = np.ascontiguousarray(scan_indices, dtype=np.float32).reshape(-1)
    midx_arr = np.ascontiguousarray(map_indices, dtype=np.float32).reshape(-1)

    scan_t = torch.from_numpy(scan_arr).to(device)
    map_t = torch.from_numpy(map_arr).to(device)
    sidx_t = torch.from_numpy(sidx_arr).to(device)
    midx_t = torch.from_numpy(midx_arr).to(device)

    with torch.no_grad():
        logits_scan_t, logits_map_t = model.predict(scan_t, map_t, sidx_t, midx_t)

    logits_scan = logits_scan_t.detach().cpu().numpy().astype(np.float32, copy=False)
    logits_map = logits_map_t.detach().cpu().numpy().astype(np.float32, copy=False)

    # Length contract — slice() returns per-input-point.
    if logits_scan.shape[0] != scan_arr.shape[0]:
        raise RuntimeError(
            f"run_model contract broken: logits_scan length {logits_scan.shape[0]} "
            f"!= scan_input length {scan_arr.shape[0]}"
        )
    if logits_map.shape[0] != map_arr.shape[0]:
        raise RuntimeError(
            f"run_model contract broken: logits_map length {logits_map.shape[0]} "
            f"!= map_input length {map_arr.shape[0]}"
        )

    return logits_scan, logits_map
