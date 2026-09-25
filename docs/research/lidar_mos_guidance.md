# LiDAR Moving-Object Proposals — Alignment Guidance (Chen et al. + UniLiPs IWU)

**Papers**:
- Chen, Mersch, Nunes, Marcuzzi, Vizzo, Behley, Stachniss, "Automatic Labeling
  to Generate Training Data for Online LiDAR-based Moving Object
  Segmentation", arXiv 2201.04501 (RA-L 2022).
- UniLiPs, arXiv 2601.05105 — §3's Iterative Weighted Update (`f_IWU`,
  Eqs. 2–4) and its "no correspondence in the refined map" moving rule. (Its
  Eq. 1 occlusion test is covered in `semantic_lifting_design.md`.)

**Component**: `lidar_preprocessing`, Steps E (`iwu/`) and F
(`motion_proposals/`). **Status: implemented.**

Both papers produce moving-object labels offline, with no learned model, from
geometry alone. We use them to add a *recall-oriented proposal* artifact to
lidar_preprocessing. False positives in it are acceptable, because
downstream association (proposal_generation, tracking) removes them cheaply.
The model-free AW log-odds split stays the default `--seg aw`.

---

## What the papers do

### Chen et al. — offline MOS auto-labeling

```
LiDAR scans ─► SLAM poses (SuMa)
            ─► coarse dynamics by map cleaning (ERASOR)
            ─► class-agnostic instances (HDBSCAN, ε ∈ {2, 1, 0.5, 0.25})
            ─► multi-object tracking (EKF, constant velocity, Hungarian)
            ─► instance is MOVING iff its trajectory > its box's max side
            ─► every point inside a moving box labelled moving
```

- Association cost (Eqs. 4–7): `C = α_d·‖c_i−c_j‖ + α_o·(1−IoU) + α_v·(1−min(v)/max(v))`,
  all α = 1. A new track is started if any term exceeds its gate
  (`T_d = 2 m, T_o = 0.95, T_v = 0.7`). Deactivated tracks are kept for
  `n_old = 5` frames.
- Instances smaller than `N_min = 5` points, or longer than `T_size = 20 m`, are dropped.
- Table I (IoU_MOS on SemanticKITTI): Octomap-style 13.6, Removert 15.7,
  ERASOR 19.1, **full pipeline 74.2**. Most of the gain comes *after* the
  coarse stage.

### UniLiPs — Iterative Weighted Update

Every map point `m` carries a static probability `P(m)`, updated per scan:

```
reinforce (Eq. 3):  P ← α·P + (1−α)·r*·(1+C(m))       scan point within 30 cm
decay     (Eq. 4):  P ← α·P + (1−α)·(1−r*)·(1−C(m))   otherwise
r* = min(1, r_max / r),  r_max = 200 m;  C(m) = label consensus;  α = 0.7, τ_s = 0.5
```

The paper iterates over **scan points**: each point updates its nearest map
point. Map points ending below `τ_s` are floaters and are removed. Moving
objects are then the scan points with no correspondence in the refined map,
checked over three consecutive scans. Ablation (UniLiPs Table 8): without
`f_IWU`, 3D detection mAP falls 31.0 → 11.7.

---

## What we already had

| Need | What exists | Where |
|---|---|---|
| Poses | eidos SLAM odometry via ingest | `poses.parquet` |
| Coarse map cleaning | Amanatides-Woo log-odds ray casting, with sensor-model constants, carve margin, grazing gate and range credibility | `classify/` (`--seg aw`) |
| Learned MOS (opt-in) | MF-MOS | `mf_mos/` (`--seg mos`) |
| Fusion | AW-static veto of MF-MOS + persistence motion filter | `union/` (`--seg union`) |
| Bag static map | voxel-snapped 0.30 m | `reduce/` → `global_static_map.npz` |
| Leakage metric | on-static % vs a shift-null chance floor | `scripts/compare_seg_dynamic.py` |

Two measured findings from `main` shaped the design. Both are recorded in
`union/motion_filter.py`:

1. A centroid-**velocity gate tested worse**. Per-sweep visibility drifts a
   cluster's centroid as the ego passes static structure, which fakes motion.
2. A **translating-cluster rescue** of persistent points re-admitted ~77%
   structure through **wall-sliding**.

---

## How it maps onto lidar_preprocessing

```
A deskew → B seg (aw | mos | union) → C ground → D reduce
   → E iwu  (bag)   global_static_map.npz → global_iwu.npz (p_static, evicted)
   → F motion_proposals (per chunk, parallel)
        per point:  source_bits  (7 heuristics, see below)
        per frame:  HDBSCAN on seed bits → boxes → wato_common.tracking
        per cluster: motion_score, track_life, n_sources, frac_seg_dynamic,
                     frac_persistent, box_filled  → motion_clusters.parquet
```

| bit | source | role |
|---|---|---|
| `AW_DYNAMIC` | AW carved-dynamic voxel | seed |
| `AW_AMBIGUOUS` | AW voxel between the two p_occ thresholds (a new export) | attach |
| `IWU_EVICTED` | near an IWU floater | seed |
| `MF_MOS` | raw MF-MOS mask (if it ran) | seed |
| `SEG_DYNAMIC` | the seg method's final `dynamic_mask` | seed |
| `BOX_FILL` | inside a moving cluster's box (Chen) | — |
| `UNMAPPED` | no IWU-refined map point nearby (UniLiPs' moving rule) | attach |

`*_dynamic_mask.npy` is untouched and remains the precision artifact.
perception_2d's depth anchors read it (semantic_lifting carries its path for
planned dynamic-point handling). The proposals are a separate recall artifact,
and neither may be widened into the other.

---

## Coarse stage: why not ERASOR

| | AW log-odds (have) | ERASOR |
|---|---|---|
| Signal | 3D free space along every ray; a voxel is dynamic if it was hit and later seen through | 2.5D height span per polar bin, scan vs map |
| Resolution | voxel (0.25 m) | bins several metres wide at range |
| Fails on | pose noise (mitigated: carve margin, grazing gate) | overhangs, trees, bridges; needs a ground revert |
| Chen Table I alone | 13.6 (untuned Octomap-style) | 19.1 |

Both detect the same event: space occupied at one time and empty at another.
Their recall overlaps, and ERASOR is the coarser of the two. Chen's own
numbers show the post-coarse stage matters more (19 → 74). The complementary
signal worth adding was scan-vs-map range-image visibility, which IWU (Step E)
supplies. Revisit ERASOR only if a measured recall gap remains.

Blind spot shared by every map-cleaning method: a vehicle moving at ego speed
is carved only through ground or background rays that cross its vacated space.

---

## Deviations from the papers

| Paper | Ours | Why |
|---|---|---|
| IWU reinforces each scan point's *nearest* map point | Every map point with a return within the match radius | The map is voxel-snapped at the radius pitch (0.30 m). "Nearest" starved between-ring map points, and on real data got them evicted. |
| IWU decays a scan point's nearest map point when it is > 30 cm away | Decay only if every return passing within ρ of the map point ended beyond it (seen through) | The literal rule lets a pedestrian's returns erode the wall behind them, and cannot tell occluded from gone. |
| — | ρ = match radius + half-sweep sensor travel; window = atan(ρ/r) + 1 ring/column; no decay inside ≈ 9 m | World NPZs keep one mid-sweep origin, and world-aligned image rows alias against the scanner's rings (measured below). |
| r* uses r_max = 200 m | `SensorModel.range_weight` (d* = voxel / divergence ≈ 100 m) | One range-credibility model for the whole component. |
| C(m) = label consensus | C = 0 (hook: `consensus=`) | No semantics in this stage yet. |
| Chen relabels non-moving clusters static | Soft features only; nothing dropped | This is a proposal source; downstream thresholds. |
| Chen's coarse stage: ERASOR | Union of AW / IWU / MF-MOS / seg / unmapped bits | See above. |
| Chen: moving iff trajectory > max side | `motion_score` = **net** displacement of the Kalman-**filtered** centre (median of first/last 3 frames) ÷ max side | Path length accumulates visibility jitter, which is the velocity-gate failure measured on `main`. |
| Chen box fill: every moving box | Only if `track_life ≥ 3` and `frac_persistent < 0.5`; box grown down to the ground | Wall-sliding fragments pass the size-normalised test (measured below). Growing the box down restores wheels and feet removed by the 0.25 m height floor. |
| HDBSCAN with ε ∈ {2, 1, 0.5, 0.25} | sklearn HDBSCAN (it integrates over ε itself) on a 0.1 m-deduplicated seed grid | Same clusters, bounded runtime. |

---

## Measured on real data

**nuScenes scene-0061**: one 19 s chunk, 382 HDL-32E sweeps, `--seg union`,
191k global map points. There are no labels, so the precision proxies are:

- **Enrichment**: how much more likely IWU eviction is within 0.5 m of the
  union dynamic cloud than elsewhere.
- **On-static leakage above chance**: the `compare_seg_dynamic` metric.

**IWU variants** (77 sweeps sampled at 4 Hz):

| Rule | Map evicted | Enrichment |
|---|---|---|
| Single-pixel seen-through, nearest-only reinforce | 54.4% | 1.3× |
| Fixed 3×3 min-filter window, nearest-only reinforce | 25.1% | 1.7× |
| Fixed 5×9 window, nearest-only reinforce | 9.1% | 2.7× |
| ρ-window (no aliasing term), reinforce-all | 14.1% | 1.6× |
| **ρ-window + 1 ring/column, reinforce-all (shipped)** | **7.0%** | **2.0×** |

The first rule is the paper's decay condition tested per pixel. The fixed 5×9
window reaches higher enrichment (2.7×) at slightly more eviction (9.1%), but
its size was picked for this scanner and scene. The shipped rule's window
follows from ρ and the scanner geometry, and it evicts the least.

**Step F** (shipped rule):
- 5.4% of points carry a bit.
- 21.6k clusters (57 per frame); 404 score > 1.
- 12 tracks are box-filled, all below the `frac_persistent < 0.5` gate by
  construction. Of the ten longest moving tracks, the eight filled ones have
  `frac_persistent ≤ 0.11`, and all but one have seg-dynamic or MF-MOS support.
- Two persistent "movers" in that list are left unfilled: a
  0.5 × 0.2 × 1.3 m track (persistence 0.92; pole- or person-sized), and a
  1.0 × 0.5 × 1.2 m track (0.52, `motion_score` 2.65). Either may be a real,
  slow pedestrian — there are no labels to tell. That is the recall cost of
  the gate, and it affects box fill only: both cluster rows keep their scores.

On-static leakage above chance, with the IWU-refined map as reference:

| Group | Leak (pts) |
|---|---|
| IWU_EVICTED | +20.6 |
| BOX_FILL | +21.4 |
| moving clusters | +25.6 |
| SEG_DYNAMIC | +31.2 |
| MF_MOS | +55.7 |
| AW_DYNAMIC | +64.1 |
| AW_AMBIGUOUS | +72.7 |
| UNMAPPED | −10.1 (off-map by definition) |

AW bits dominate the union's leakage, as the `union` notes predict. Downstream
should weight by `n_sources`, `frac_persistent` and `motion_score`, not by any
single bit.

**Runtime**: IWU 9 s and Step F 47 s for this chunk. The single-pixel
variant's Step F took 113 s, because it produced more clusters.

---

## How downstream should consume it

- **proposal_generation**: each `motion_clusters.parquet` row is a
  `ProposalRow` with `provenance="lidar_mos"`; the box columns share names.
  Useful gates are `motion_score > 1`, `n_sources ≥ 2` and
  `frac_persistent < 0.5`. Rows with a class-less box still need a class from
  semantic_lifting's labels.
- **tracking**: build on `wato_common.tracking` (Chen cost, Kalman, Hungarian)
  rather than a second motion model. `track_hint_id` is chunk-local and **not**
  an identity.
- **perception_2d (future SAM2 LiDAR prompts)**: project moving-cluster points
  (`cluster_id` in rows with `motion_score > 1`), not raw `source_bits`.
- **Exclusion uses** (depth anchors, static maps): keep using `dynamic_mask`.

---

## Gaps and open questions

1. **Ground-truth recall.** Build nuScenes MOS ground truth from annotated
   boxes (points inside boxes with speed > 0.5 m/s) and report recall and
   IoU_MOS (Chen Eq. 9) per bit. The on-static metric only measures leakage,
   not what is missed. This is also the evidence `sensor_model.py` says would
   move its probabilities.
2. **WATO rig.** Every number above is from nuScenes HDL-32E. Re-measure on a
   3-Velodyne bag with `.wato.yaml`. That bag has per-frame fusion, VLP-16
   corners, and a sweep rate of 20 Hz.
3. **Per-point ray origins.** These would let IWU judge points within 9 m and
   shrink ρ. The same follow-up applies to the AW carve.
4. **Semantic C(m).** Feed semantic_lifting's accumulated `(label, count)`
   into IWU's `consensus` (the full UniLiPs form).
5. **AW_DYNAMIC as a seed.** It is the largest and leakiest seed. If
   downstream finds its clusters too costly, make it attach-only (a one-line
   change in `SEED_BITS`) and re-measure recall with (1).
6. **Coherence gate.** Consolidate `union`'s greedy linker onto
   `wato_common.tracking` after re-measuring it with `compare_seg_dynamic`.

---

## Summary of actionable steps

Done:
- `iwu/`: hybrid IWU with a windowed seen-through test, and `global_iwu.npz`.
- `motion_proposals/`: source bits, HDBSCAN, tracking features, persistence-gated
  box fill; `motion_clusters.parquet`.
- `wato_common/tracking/`: `fit_bev_box`, `iou_3d`, `BoxKalmanFilter`,
  `associate`, `MultiObjectTracker`.
- `classify` exports `ambiguous_voxel_keys`.
- `range_image.py` is shared by `mf_mos/` and `iwu/`.
- `--two-pass` is off by default and renamed as a global-map prior. It is not IWU.
- `compare_seg_dynamic.py --proposals`, and `viz --layer proposals`.

Next:
- Gaps 1–2 above (ground-truth recall, WATO rig).
- proposal_generation consuming `motion_clusters.parquet`.
- tracking built on `wato_common.tracking`.
