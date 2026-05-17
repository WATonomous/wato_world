# Cross-Modal Uncertainty Bookkeeping Alignment Guidance

**Papers synthesized**:
- "HINTED: Hard Instance Enhanced Detector with Mixed-Density Feature Fusion
  for Sparsely-Supervised 3D Object Detection" (Xia et al., CVPR 2024,
  arXiv 2404.00114) — uncertainty-aware pseudo-label fusion
- "MS3D++" (Tsai et al., 2024) — per-detector confidence calibration
- "Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection"
  (Karner et al., ITSC 2022) — pattern for cross-modality disagreement signals

The four "biggest accuracy wins" (detector ensemble, SLF, DA-pseudo-LiDAR,
body-frame aggregation) all produce a 3D box, but with very different
failure modes. The cross-modal uncertainty bookkeeping turns that diversity
into a quality signal: per-proposal disagreement is a strong predictor of
which boxes need extra refinement and which are already trustworthy.

This is small in code surface area but large in downstream leverage —
`label_refinement` reads these fields and weights its training and inference
accordingly. The user has elected (per Phase plan) to make this *diagnostic
columns only* in v1: no active gating, no dual code paths. Just record the
numbers so future iterations can learn from them.

---

## What the uncertainty fields express

```
   per proposal (one row in proposals.parquet):

   ┌─────────────────────────────────────────────────────┐
   │ n_detectors_agreeing  ∈ {0, 1, 2, 3}                │  ← MS3D++ ensemble
   │   higher = stronger LiDAR-side agreement            │
   ├─────────────────────────────────────────────────────┤
   │ ensemble_score_var    ∈ [0, ∞)                       │  ← std of detector scores
   │   lower = detectors agree on confidence              │
   ├─────────────────────────────────────────────────────┤
   │ slf_dice_loss         ∈ [0, 1]                       │  ← SLF L_mask at convergence
   │   lower = SDF reprojection matches SAM2 well         │
   ├─────────────────────────────────────────────────────┤
   │ slf_lidar_chamfer     ∈ [0, ∞) metres                │  ← SLF L_lidar at convergence
   │   lower = SDF surface matches LiDAR points well      │
   ├─────────────────────────────────────────────────────┤
   │ slf_depth_chamfer     ∈ [0, ∞) metres                │  ← SLF L_depth at convergence
   │   lower = SDF surface matches DA pseudo-LiDAR well   │
   ├─────────────────────────────────────────────────────┤
   │ lidar_density_in_box  ∈ [0, ∞) points / m³           │  ← raw LiDAR support
   │   higher = denser observation, more reliable         │
   ├─────────────────────────────────────────────────────┤
   │ da_pixels_in_mask     ∈ [0, ∞) integer count          │  ← DA pseudo-LiDAR support
   │   higher = stronger camera depth evidence            │
   └─────────────────────────────────────────────────────┘
                                │
                                ▼
   uncertainty  ∈ [0, 1]    weighted combination (see formula below)
                            0 = high confidence, 1 = low
```

The combined `uncertainty` score is what downstream code consumes most often.
The individual fields are kept so that future analysis can attribute
uncertainty to a specific failure mode.

---

## Combination formula (v1)

This is a hand-tuned linear combination. The intent is not perfect
calibration — that needs a labeled dataset — but a useful ordering signal.

```python
def compute_uncertainty(prop: ProposalRow, n_detectors: int) -> float:
    # Normalize each signal to [0, 1] where 0 = good.
    u_ensemble = 1.0 - (prop.n_detectors_agreeing or 0) / max(n_detectors, 1)
    u_score_var = min((prop.ensemble_score_var or 0.0) / 0.2, 1.0)
    u_slf_dice  = prop.slf_dice_loss or 1.0   # already [0, 1]
    u_slf_lidar = min((prop.slf_lidar_chamfer or 1.0) / 0.5, 1.0)
    u_slf_depth = min((prop.slf_depth_chamfer or 1.0) / 0.5, 1.0)
    # Density: invert and clamp so dense = 0, sparse = 1
    density_class_prior = {"vehicle": 50.0, "pedestrian": 200.0, "cyclist": 100.0}.get(prop.cls, 50.0)
    u_density = 1.0 - min((prop.lidar_density_in_box or 0.0) / density_class_prior, 1.0)

    return float(np.clip(
        0.30 * u_ensemble +
        0.10 * u_score_var +
        0.20 * u_slf_dice +
        0.15 * u_slf_lidar +
        0.10 * u_slf_depth +
        0.15 * u_density,
        0.0, 1.0,
    ))
```

**Why these weights**: ensemble agreement is the strongest signal — three
independent detectors agreeing is harder evidence than any single mask loss.
SLF dice loss is next: if the SDF can't be reprojected to match the SAM mask
at convergence, the box is wrong. LiDAR chamfer and density tie because
"close fit on few points" and "loose fit on many points" are both diagnostic
of different problems.

**Calibration plan (deferred to v2)**: train an isotonic regression mapping
this raw score to `P(IoU > 0.5 | uncertainty)` using a labeled subset.
Until then, treat `uncertainty` as ordinal not probabilistic.

---

## How label_refinement consumes these fields

Per the Phase plan, v1 only records the diagnostics. `label_refinement` reads
them and uses them as **soft weights** during training and inference:

- **Inference**: lower-uncertainty proposals get higher weight in the
  LabelFormer pose head — a confident proposal pulls the refined trajectory
  toward itself. High-uncertainty proposals are still used but contribute
  less.
- **Training (when we have GT)**: uncertainty becomes the sample weight in
  the loss. Low-uncertainty pseudo-labels are trusted; high-uncertainty
  ones contribute proportionally less.

The exact integration is a small change inside `label_refinement/model.py`'s
pose head: multiply the per-frame attention weight by `(1 - uncertainty)`
before softmax-normalization.

---

## What we already have

| Uncertainty field | Source | Where computed |
|---|---|---|
| `n_detectors_agreeing` | MS3D++ KBF cluster size | `detectors/ensemble.py` |
| `ensemble_score_var` | std over detector scores in cluster | `detectors/ensemble.py` |
| `slf_dice_loss` | Adam fitter final L_mask | `slf/fitter.py` |
| `slf_lidar_chamfer` | Adam fitter final L_lidar | `slf/fitter.py` |
| `slf_depth_chamfer` | Adam fitter final L_depth | `slf/fitter.py` |
| `lidar_density_in_box` | count(points) / volume(box) on dynamic_pts | `proposal_generation/pipeline.py` |
| `da_pixels_in_mask` | mask pixel count post-clamp | `proposal_generation/pseudo_lidar.py` |
| `uncertainty` | combination of the above | `proposal_generation/uncertainty.py` |

All upstream signals already exist or are being computed for other reasons.
This component is pure bookkeeping — we collect what's already there.

---

## Gaps and what to do

### 1. Extend ProposalRow / PROPOSAL_SCHEMA

Already covered in Phase 1 shared infrastructure changes. The seven optional
float/int columns above must be added to both `ProposalRow` (Pydantic) and
`PROPOSAL_SCHEMA` (PyArrow).

### 2. Wire computation into proposal_generation pipeline

In `proposal_generation/pipeline.py`'s per-chunk function, after the SLF
fitter and ensemble fusion complete, fill in each row's diagnostic fields
before calling `wato_common.io.parquet_io.write_table()`. This is where
`uncertainty.compute_uncertainty()` is called once per row.

### 3. Per-class density priors

The class-prior density values `{"vehicle": 50.0, "pedestrian": 200.0,
"cyclist": 100.0}` (points / m³) are empirically reasonable for nuScenes /
KITTI but should be measured on the user's recording rig once a labeled bag
is available. Make them configurable via
`proposal_generation.yaml:uncertainty.class_density_priors`.

### 4. Reading + using uncertainty in label_refinement

In `label_refinement/model.py`'s pose head:

```python
# Existing:  pose_logits = self.pose_head(features)
# New:
weights = 1.0 - track_uncertainty            # (T,) per frame, from proposals
pose_logits = self.pose_head(features)        # (T, D)
weighted = pose_logits * weights[:, None]     # downweight high-uncertainty frames
```

For the size head (one output per track), use the *median* uncertainty
across the track's frames as a single weight.

### 5. Diagnostic outputs for analysis

Optionally write `proposal_diagnostics.parquet` per chunk (separate from
`proposals.parquet`) containing the unbucketed per-stage signals
(e.g., individual detector scores, per-camera SLF visibility flags, raw
ICP residuals from aggregation). Lets researchers post-hoc analyse what
the ensemble + SLF + DA actually said before fusion compressed it. Out of
scope for v1; documented here as a future iteration.

---

## Summary of actionable steps

1. Extend `ProposalRow` / `PROPOSAL_SCHEMA` with the seven diagnostic columns
   (Phase 1).
2. Build `proposal_generation/uncertainty.py` with `compute_uncertainty()`
   implementing the formula above.
3. Wire `compute_uncertainty()` into `proposal_generation/pipeline.py` right
   before parquet write.
4. Add `uncertainty.class_density_priors` knob to
   `proposal_generation.yaml` so the prior is configurable per dataset.
5. In `label_refinement/model.py`, use `uncertainty` as a per-frame weight
   in the pose head and as a per-track weight in the size head (when
   ingesting from `proposals.parquet` via `tracks.parquet` provenance).
6. Unit test: golden ProposalRow with known signal values → assert
   `compute_uncertainty()` returns the expected scalar within tolerance.
7. (Future) Add calibration recipe and `proposal_diagnostics.parquet`.
