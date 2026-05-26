# Segment-Lift-and-Fit (SLF) Alignment Guidance

**Paper**: "Segment, Lift, and Fit: Automatic 3D Shape Labeling from 2D Prompts"
**Reference**: themoonlight.io/en/review/segment-lift-and-fit-automatic-3d-shape-labeling-from-2d-prompts

SLF turns per-camera 2D instance masks into full 3D bounding boxes (position,
orientation, and shape) using a gradient-descent fitting loop.  It is the
primary design for the `proposal_generation` component's geometry side
(complementing the LiDAR detector side).

---

## What SLF does

```
2D prompt (point / box)
       │
       ▼
  ┌─────────┐
  │  STAGE 1│  SAM → per-camera 2D instance mask
  │ SEGMENT │
  └────┬────┘
       │  2D mask (H×W bool)
       ▼
  ┌─────────┐
  │  STAGE 2│  Lift mask into 3D via SDF + PCA shape space
  │  LIFT   │  shape latent z ∈ R^k  (k=10 or 20 PCA components)
  └────┬────┘     shape prior built from dataset of 3D vehicle meshes
       │  initial (pose, shape) in vehicle body frame
       ▼
  ┌─────────┐
  │  STAGE 3│  Adam optimizer minimizes:
  │   FIT   │    L_mask  = dice loss (projected SDF mask vs. SAM mask)
  └────┬────┘    L_lidar = Chamfer/surface loss (SDF surface vs. LiDAR pts)
       │         L_ground = z-alignment to ground plane height
       │         (SE(3) pose + latent z as free variables)
       ▼
  3D bounding box  (center x,y,z · W,L,H · heading θ)
```

**~90% AP on KITTI bounding box annotations**; detectors trained on SLF
pseudo-labels match performance of detectors trained on GT labels.

---

## What we already have that SLF needs

| SLF requirement | Our artifact | Where |
|-----------------|-------------|-------|
| Per-camera 2D masks | `masks_2d/` from `perception_2d` | `perception_2d` output |
| Camera intrinsics K | `calibration.json` → `cameras[cam_id].K` | ingest output |
| Camera extrinsics | `calibration.json` → `ego_T_cam` (via TF chain) | ingest output |
| LiDAR world-frame points | `world/*.npz` from `lidar_preprocessing` | lidar_preprocessing output |
| Dynamic object points | `dynamic_masks/*.npy` (per-sweep bool mask) | lidar_preprocessing output |
| Ground plane height | `ground.npz` → `height_grid`, `grid_origin`, `cell_size` | lidar_preprocessing output |
| Surface normals | `ground.npz` → `normal_grid` | lidar_preprocessing output |

The `ground.py` step was built precisely to produce the height grid and surface
normal grid that the SLF ground alignment loss term consumes.

---

## How our multi-sensor rig improves on the paper

**12 cameras vs. KITTI's 2**

SLF's mask alignment loss `L_mask` is a sum over cameras.  With 12 cameras
(4 lower directional + 8 panoramic), we get:
- Objects visible from multiple angles simultaneously → stronger constraint
  on pose (heading angle) and shape (which face is which)
- Panoramic cameras capture objects at oblique angles not covered by
  forward-facing cameras → better W/L disambiguation
- Lower cameras see the side of pedestrians and cyclists from close range →
  better H estimation for non-vehicle classes

**3 LiDARs vs. KITTI's 1**

SLF's LiDAR surface loss `L_lidar` uses the closest LiDAR returns to the
fitted SDF surface.  With three Velodynes (center + NE + NW):
- 3× point density on average → denser surface sampling → stronger gradient signal
- NE/NW LiDARs see occluded sides of objects that the center LiDAR misses
- Multi-LiDAR coverage also reduces the degenerate case where an object sits
  in the blind spot of the single center LiDAR

---

## Gaps and what to do

### 1. SAM3 masks from perception_2d (prerequisite)

SLF Stage 1 is SAM.  `proposal_generation` must receive per-camera, per-object
2D masks from `perception_2d`.  The contract is:

```
masks_2d/<chunk_id>/<cam_id>/<frame_seq>/<detection_id>.png  (binary mask)
detections_2d.parquet  (bbox, class, score, cam_id, frame_seq, detection_id)
```

`proposal_generation` reads these and groups detections by 3D candidate
position (using LiDAR-projected depth to get an approximate world location
for each 2D detection).

### 2. Vehicle shape prior (PCA over 3D meshes)

**What SLF does**: builds a low-dimensional shape latent space by running PCA
over a library of 3D vehicle CAD meshes, then represents each shape as a signed
distance function (SDF) in that latent space.

**What to build**:
- Collect vehicle mesh libraries: ShapeNet (cars), nuScenes object assets,
  or KITTI ground-truth box shapes.
- Voxelize each mesh into a 3D SDF grid (e.g. 32³ or 64³ resolution).
- Flatten and run PCA to get a basis of k=10–20 components.
- Store `shape_prior.npz` containing (mean_sdf, components, explained_variance).
- At fit time, the latent shape vector `z ∈ R^k` expands via
  `sdf_grid = mean + components.T @ z`.

Need one shape prior per semantic class (vehicle, pedestrian, cyclist).
Pedestrian and cyclist SDF priors may benefit from a small k (3–5) because
human body shape is more constrained than vehicle shape.

### 3. Differentiable SDF renderer

**What SLF does**: for each camera and each optimized pose + shape, renders
the SDF as a 2D mask by raycasting through the SDF field.  The dice loss
between the rendered mask and the SAM mask provides gradients for Adam.

**What to build in `proposal_generation`**:
- A `sdf_renderer.py` that, given `(R, t, z)` (pose + latent shape) and a
  camera `(K, cam_T_world)`, raycasts the SDF and returns a binary or soft
  occupancy mask in image space.
- Implement in PyTorch so autograd provides gradients for the optimizer.
- Consider starting with a simpler ellipsoid SDF (no learned PCA) as a
  scaffold — it reduces a 3D box to just (cx, cy, cz, w, l, h, θ) and the
  renderer is a closed-form signed-distance projection.

### 4. Fitting loop

**What to build in `proposal_generation`**:

```python
def fit_proposal(
    masks_per_cam: dict[str, np.ndarray],   # cam_id -> (H, W) bool mask
    lidar_pts: np.ndarray,                  # (N, 3) dynamic LiDAR points near candidate
    ground_height: float,                   # from height_grid at (cx, cy)
    shape_prior: ShapePrior,
    K_per_cam: dict[str, np.ndarray],
    cam_T_world_per_cam: dict[str, np.ndarray],
) -> BoxProposal:
    # 1. Initialize pose from LiDAR cluster centroid + ground height.
    # 2. Initialize z = zeros (mean shape).
    # 3. Adam(lr=0.01) over (cx, cy, cz, log_w, log_l, log_h, sin_θ, cos_θ, z).
    # 4. Loss = L_mask + λ1*L_lidar + λ2*L_ground.
    # 5. Return best box after convergence (<100 gradient steps in paper).
```

The paper runs one optimizer instance per detected object, which is
embarrassingly parallel — use `concurrent.futures.ProcessPoolExecutor` (already
used in `lidar_preprocessing/pipeline.py`) to fit multiple objects in parallel.

### 5. Multi-camera mask aggregation

When the same object is visible in multiple cameras, aggregate their mask
alignment losses:
```
L_mask = (1/|cams|) * Σ_{cam ∈ visible_cams} dice(render(pose, z, cam), mask_cam)
```

Visibility check: project the box center into each camera and check if the
projected point falls within the image frame and the depth is positive.  With
12 cameras this is cheap and greatly increases fitting robustness.

### 6. Fusion with LiDAR detector proposals

SLF produces shape-accurate boxes from 2D masks.  A LiDAR-only detector
(CenterPoint or similar) produces accurate position + orientation but noisier
shape.  The paper suggests fusing both:

- Run LiDAR detector on the dynamic point cloud (from `dynamic_masks/*.npy`
  applied to `world/*.npz`).
- For each LiDAR proposal, check if a matching SLF proposal exists (IoU > 0.3
  in bird's-eye view).
- If match: use SLF shape, LiDAR position (LiDAR has better position accuracy).
- If no match: use LiDAR proposal as-is.
- Output to `proposals.parquet`: `(center_xyz, wlh, heading, score, source)`.

---

## Summary of actionable steps

1. Define `proposals.parquet` schema in `src/common/src/wato_common/schemas.py`.
2. Add `masks_2d/` artifact path to `src/common/src/wato_common/artifact_store.py`.
3. Build `shape_prior.py` in `proposal_generation`: load mesh library, voxelize
   SDF grids, run PCA, save `shape_prior.npz`.
4. Build `sdf_renderer.py` in `proposal_generation`: differentiable PyTorch
   raycast renderer for a shape-prior SDF.
5. Build `slf_fitter.py` in `proposal_generation`: Adam fitting loop as above.
6. Build `lidar_detector.py` in `proposal_generation`: CenterPoint inference
   on dynamic points per chunk.
7. Build `fusion.py` in `proposal_generation`: merge SLF + LiDAR proposals.
8. Wire into `pipeline.py` with the same chunk-parallel worker pattern as
   `lidar_preprocessing`.
