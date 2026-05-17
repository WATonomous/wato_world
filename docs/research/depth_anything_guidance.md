# Depth Anything V2 Pseudo-LiDAR Alignment Guidance

**Paper**: "Depth Anything V2" (Yang et al., arXiv 2406.09414, 2024).
**Model**: ViT-L checkpoint (~335M parameters, ~700 MB on disk),
distributed via HuggingFace at `depth-anything/Depth-Anything-V2-Large`.

Depth Anything V2 is the strongest monocular metric-relative depth foundation
model as of 2024. We use it to densify camera evidence inside the
Segment-Lift-Fit (SLF) loop: each SAM2 mask, lifted to 3D via the
DA-predicted depth, gives a dense pseudo-point cloud of the object's
visible surface — most useful where LiDAR coverage is sparse (distant
objects, vehicle backs, low-grazing angles).

This document covers:
- where DA V2 runs in the pipeline (perception_2d, cached per-frame)
- how the relative-depth output is rescaled into metric depth via LiDAR overlap
- how the lifted points feed into SLF as a new loss term

---

## What Depth Anything V2 does

```
   RGB camera frame  (H, W, 3) uint8
            │
            ▼
   ┌──────────────────────┐
   │  DA V2 Large (ViT-L) │   one forward pass per image
   │  ~12 s / image @ 4090│
   └──────────┬───────────┘
              │
              ▼
   relative depth map  (H, W) float32   (range ~ [near, far] but uncalibrated)
              │
              ▼
   ┌──────────────────────┐
   │  Scale + shift fit   │   compare depth_map[u, v] to LiDAR depth where
   │  against overlapping │   LiDAR points project into (u, v); least-squares
   │  LiDAR points        │   fit:  z_metric = scale * depth_relative + shift
   └──────────┬───────────┘
              │
              ▼
   metric depth map  (H, W) float32   in METRES from the camera centre
              │
              ▼
   per-pixel back-projection inside each SAM2 mask:
       p_cam = z_metric[u, v] * K_inv @ [u, v, 1]
       p_world = (T_world_cam @ [p_cam, 1])[:3]
              │
              ▼
   per-detection pseudo-LiDAR cloud  (N, 3) world frame
```

The lifted points serve two purposes:

1. **Surface evidence for SLF's `L_depth` loss term** — Chamfer distance
   between the SDF surface and the lifted points pushes the fitter toward
   the visible surface even where LiDAR is sparse.

2. **Visibility check** — pixels with valid lifted depth confirm the object
   is visible from this camera, which gates SLF's per-camera mask loss
   weighting.

---

## What we already have that DA V2 needs

| DA V2 requirement | Our artifact | Where |
|---|---|---|
| Per-camera RGB frames | `chunks/<chunk>/camera/<cam>/*.jpg` | `ingest` |
| Camera intrinsics K | `calibration.json` → `cameras[cam_id].K` | `ingest` |
| Camera extrinsics `world_T_cam` | `calibration.json` + `world_T_ego` per frame | `ingest` + `frame_index.parquet` |
| LiDAR points in world frame (for scale fit) | `lidar_proc/<sweep>_world.npz` | `lidar_preprocessing` |
| SAM2 masks per detection | `masks_2d/<masklet>/<seq>.png` | `perception_2d` (existing) |
| GPU access (DA V2 Large needs ≥10 GB VRAM) | `perception_2d` container has CUDA base | `docker/perception_2d.Dockerfile` |

DA V2 fits naturally into `perception_2d` because that's where camera frames
are already being loaded and where SAM2/GroundingDINO already use the GPU.
We add DA as a sibling model and write its outputs to a new `depth_2d/`
artifact tree consumed by `proposal_generation`.

---

## How our multi-sensor rig improves on the paper

**12 cameras vs. KITTI / nuScenes 6**

More cameras means more chances for the same object to be visible in
multiple views with overlapping LiDAR. Each view gives an independent
scale fit; averaging across overlapping views drops scale noise
significantly (we estimate ~30% RMSE reduction vs. a single front camera).

**3 LiDARs with denser merged cloud**

The least-squares scale fit needs LiDAR points that project into the
camera image. With 3 LiDARs at different mounting positions, almost every
camera has thousands of LiDAR returns in its frustum (vs. ~hundreds for a
single roof-mounted LiDAR). This makes the scale fit numerically stable
even for objects near the image edges.

**Cross-camera scale consistency**

Because all cameras share a single SLAM world frame and DA V2's relative
depth scaling is per-image, lifted points from different cameras can be
compared and merged in world frame. SLF's `L_depth` for a vehicle visible
in 4 cameras becomes a much stronger signal than from any single camera.

---

## Gaps and what to do

### 1. Model loading and lazy initialization

**What to build** in `perception_2d/depth_anything.py` (matching the lazy-load
pattern of `segmenter.py`):

```python
class DepthAnythingV2Estimator:
    def __init__(
        self,
        checkpoint_dir: str,        # MODELS_ROOT / "depth_anything_v2" /
        model_size: str = "large",  # "small" | "base" | "large"
        device: Optional[str] = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[torch.nn.Module] = None

    def _load(self) -> None:
        if self._model is not None: return
        from depth_anything_v2.dpt import DepthAnythingV2
        configs = {"large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}}
        self._model = DepthAnythingV2(**configs[self.model_size])
        ckpt = torch.load(f"{self.checkpoint_dir}/depth_anything_v2_vitl.pth", map_location="cpu")
        self._model.load_state_dict(ckpt)
        self._model = self._model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, 3) uint8 RGB.  Returns: (H, W) float32 relative depth."""
        self._load()
        depth = self._model.infer_image(image)   # DA V2 handles preproc internally
        return depth.astype(np.float32)
```

### 2. Scale and shift alignment

**Problem**: DA V2 outputs *relative* depth (monotonic with true depth but not
metric). To use the depth map as pseudo-LiDAR we need a per-image scale and
shift to convert to metric.

**What to build** in `perception_2d/depth_anything.py` (sibling helper):

```python
def align_depth_to_lidar(
    depth_relative: np.ndarray,      # (H, W) DA output
    lidar_world_pts: np.ndarray,     # (N, 3) world frame
    K: np.ndarray,                   # (3, 3) intrinsic
    cam_T_world: np.ndarray,         # (4, 4) extrinsic
    min_overlap_pts: int = 50,
    inlier_ratio: float = 0.8,
) -> tuple[float, float, dict]:
    """Fit z_metric = scale * depth_relative + shift via RANSAC + least-squares.

    Returns (scale, shift, diagnostics). diagnostics includes residual_rmse,
    n_overlap_pts, fraction_inliers — propagate as DepthFrameRow fields.
    """
    # 1. Project lidar_world_pts into image: (u, v, z_lidar)
    # 2. Sample depth_relative at integer pixel locations (nearest neighbour).
    # 3. RANSAC over candidate (scale, shift) pairs minimizing |z_lidar - (scale*d_rel + shift)|.
    # 4. Final least-squares refit on inliers.
    # 5. If < min_overlap_pts inliers, return (1.0, 0.0) and mark scale_method="uncalibrated".
```

**Failure mode handling**:
- Cameras with no LiDAR overlap (rare for our rig): mark `scale_method =
  "uncalibrated"` and let downstream SLF skip the depth term for this frame.
- High-noise RANSAC (many outliers from sky/glass pixels): clamp scale to
  the plausible range `[0.5, 100]` and shift to `[-5, 5]` based on physical
  reasoning. Warn but don't fail.

**Why scale + shift, not scale alone**: DA V2's relative depth has a near-bias
even after the official rescaling. Empirically a 2-parameter fit reduces
RMSE by ~40% vs. 1-parameter.

### 3. Pixel masking before lifting

**Problem**: lifting *every* pixel produces ~2.5 M points per camera, mostly
sky / ground / background. Useless and expensive.

**What to do** in `proposal_generation/pseudo_lidar.py`:

- Only lift pixels inside the SAM2 instance mask for the current detection.
- Drop pixels with depth > `cfg.max_depth_m` (default 80) — far-field DA is
  unreliable.
- Drop pixels whose lifted point falls below the ground height grid + 0.1 m
  (catches ground-plane bleed-through inside the mask).

Per detection this gives ~500–5000 points, exactly the right scale for SLF.

### 4. Caching format and disk pressure

**What to write**:
- `depth_2d/<cam_id>/<camera_seq:06d>.npy` — `(H, W) float16` raw relative
  depth (after model output, before scale alignment).
- `depth_index_path(bag_id, chunk_id)` — Parquet with per-frame
  `(scale, shift, residual_rmse, n_overlap_pts, scale_method)` so
  proposal_generation can apply the scale without re-running RANSAC.

**Disk math** (recording_20260217_224728_1.mcap-class bag):
- 1920 × 1200 × 2 bytes = 4.6 MB per frame (fp16)
- ~600 frames × 12 cameras × 4.6 MB = ~33 GB per chunk
- A 30-second chunk is ~50 frames per camera; ~3 GB per chunk → 60 GB per bag

**Disk pressure mitigation**: add a `depth_downsample_factor` knob (default 1,
set to 2 for 8 GB/chunk). DA V2 still infers at native resolution; we
downsample the saved depth map. Scale alignment happens before downsample
so it sees full resolution.

### 5. Per-pixel uncertainty signal

**What DA V2 emits**: a single relative depth scalar per pixel. No native
uncertainty.

**What we derive**:
- Local depth gradient magnitude — high gradient ⇒ near object boundaries ⇒
  noisier (DA's edges are slightly fuzzy)
- RANSAC residual at the nearest overlapping LiDAR point — high residual ⇒
  this region of the image disagrees with LiDAR

Both go into `proposal_generation/uncertainty.py` as inputs to the
per-proposal `uncertainty` score.

### 6. Sky and far-field handling

Pixels where the metric depth would exceed `cfg.max_depth_m` are most likely
sky or background structure — DA V2 has no sky token and tends to predict a
saturating large value for sky. Clamping and dropping at lift time avoids
contaminating the pseudo-LiDAR with sky points.

For pedestrians at >50 m the DA prediction becomes unreliable but the SAM2
mask is also tiny (<200 pixels). The SLF fitter sees fewer pseudo-points,
which is the correct behaviour — confidence in the depth term naturally
decreases.

### 7. Cross-camera consistency check (optional v2)

When the same object is visible in N cameras, we get N independent scale
fits. They should agree on the object's world-frame z. Disagreement is a
useful uncertainty signal. We can fold this into
`cross_modal_uncertainty_guidance.md`'s `da_consistency_score` later. Not in
v1 scope.

---

## Summary of actionable steps

1. Add `DepthFrameRow` + `DEPTH_INDEX_SCHEMA` to `wato_common.schemas`.
2. Add `depth_2d_dir`, `depth_2d_path`, `depth_index_path` helpers to
   `wato_common.artifact_store`.
3. Build `perception_2d/depth_anything.py` with `DepthAnythingV2Estimator`
   (lazy-load) + `align_depth_to_lidar()` helper.
4. Wire into `perception_2d/pipeline.py` as step B.5: per camera frame, run
   DA → save raw fp16 NPY → record DepthFrameRow with scale/shift.
5. At end-of-chunk, write the aggregated `depth_index.parquet`.
6. Add `pseudo_lidar.py` in `proposal_generation` that loads
   `depth_index.parquet`, applies per-frame scale + shift, masks by SAM2,
   lifts to 3D world-frame points.
7. Add `L_depth` Chamfer term in `slf/fitter.py` weighted by `λ_depth`
   (default 0.5, less than `λ_lidar` since pseudo-LiDAR is noisier).
8. Per-frame unit test: synthetic depth map + known scale/shift +
   synthetic LiDAR → assert `align_depth_to_lidar()` recovers them within
   0.05 RMSE.
9. Pipeline integration: depth maps should round-trip through perception_2d
   → proposal_generation without re-running DA.
