# perception_2d

Per camera: a detector boxes keyframes, SAM2.1 with DAM4SAM memory tracks each box
into a masklet (one per physical object), DINOv2 describes each masklet's
appearance in every image, and Depth Anything V2 gives a metric depth map per
image, fitted to static LiDAR.
Cross-camera identity is left to the `tracking` component.

## Running

Fetch the weights on the host first, into `${MODELS_ROOT}` (default
`./data/models`, mounted read-only at `/data/models`):

```bash
python3 src/perception_2d/scripts/fetch_models.py [--skip depth_anything_v2]
```

Then, after ingest and lidar_preprocessing:

```bash
./watod run perception_2d --bag <bag_id> [--chunk <chunk_id>] [--force]
./watod run perception_2d viz --bag <bag_id> [--chunk <chunk_id> --cam <cam_id>]
./watod run perception_2d embed --bag <bag_id> [--chunk <chunk_id>] [--force]  # descriptors only
./watod run perception_2d ego-mask --bag <bag_id> [--config <yaml>]   # see "Ego mask"
./watod run perception_2d compare --bag <bag_id> --baseline <root>    # see "Comparing two runs"
./watod test perception_2d
```

WATO bags (`ingest.wato*.yaml`) take the rig's config, which adds the car's
ego mask: `--config /ws/src/perception_2d/config/perception_2d.wato.yaml`.

`embed` writes `masklet_embeddings.npz` for chunks already tracked, from their
masks on disk, without the detector or SAM2 (minutes per chunk): for a bag
tracked before the file existed, or after a change to the descriptor. It
leaves masks, tables and the chunk's manifest as they are.

A chunk whose output parquets exist is skipped. Each camera's masklets are
checkpointed, so an aborted run resumes per camera. `--force` discards those
caches and clears the chunk's `masks_2d/` and `depth_2d/` first (mask
directories are named by masklet id, which is new every run). A resume cache
made under another effective config or mask backend (or one recording
neither) stops the run rather than mix masks: rerun with `--force`. A finished
chunk is recomputed only with `--force`, whatever the config.

## How a chunk runs

Three passes; depth and tracking never share VRAM:

1. **Depth.** DA-V2 outputs relative *disparity*, so the fit is in inverse-depth
   space, `1/depth = scale·disparity + shift`, by RANSAC with inliers judged in
   metres. The anchors are the static (`~dynamic_mask`) world-frame points of
   every sweep the image matched, projected with the ego pose at the image's
   own `camera_timestamp_ns`, through the shared full-intrinsic/lens
   [camera model](../../docs/camera_geometry_contract.md). With too few anchors, a recent frame's fit is
   reused; a frame with no usable fit is not written.
2. **Tracking.** The detector runs every `redetect_every_k` images. A box
   matching a track (box IoU with its mask bbox ≥ `iou_match_threshold`)
   **supports** it. The rest, highest score first and less repeats of a better
   box at the same keyframe, go through the **birth gate**, which tests each
   against the masks of the live tracks and of the births already accepted at
   that keyframe (`birth_gate`):
   - `untracked_area` (default, [SAM2MOT](https://arxiv.org/abs/2504.04519)'s
     object addition): the box starts a track only if more than
     `birth_untracked_fraction` (its 0.5) of it lies outside every tracked
     mask;
   - `mask_iou` ([SAM 3](https://arxiv.org/abs/2511.16719), App. C.3): SAM2's
     image mask for the box must have IoU below `birth_max_mask_iou` (its 0.1)
     with every tracked mask; one extra image encoding per keyframe with
     births. Unlike `untracked_area` it lets a small box inside a large mask
     through;
   - `none`: every such box starts a track.

   A box on a corner of a tracked car, or on one car of a mask bled over two,
   matches no mask bbox but lies inside a tracked mask, so it is refused at any
   detector score; a refused box neither starts nor supports a track. Then,
   per track:
   - **Keep-alive.** It rises by one per supported keyframe (up to
     `support_keyframes`) and falls by one per keyframe the mask covers at
     least `support_min_area` of the image with no supporting box; at zero the
     track ends. A large mask the detector doesn't see (the sky, the road)
     ends within a few keyframes; a distant car COCO YOLO boxed once keeps its
     track. With `birth_score` > 0, a weaker box starts a track tentatively,
     its masks held back until boxes support it at `confirm_keyframes`
     keyframes in a row (off by default: YOLO rarely re-boxes distant cars).
   - **Re-prompting.** A confident box (≥ `reprompt_score`) backing one track
     alone, whose mask bbox has drifted below `reprompt_below_iou` from it,
     re-prompts that track.
   - **Class.** The one its boxes' scores sum highest for; its score is their
     mean.
   - **Identity.** A track ends when its object has had no mask for
     `drop_lost_after_frames` images. Detector support also controls its
     lifetime; overlap with another live mask does not end it.
   - **Independent masks.** Each mask drops its specks (components under
     `island_fraction` of its largest) and preserves every remaining claimed
     pixel, including pixels claimed by another object. Pixel logits,
     containment and overlap collapse do not prune competing hypotheses.
     Lifting retains these claims until proposal generation refines their
     LiDAR support and resolves ownership in 3D.
   - **Ego vehicle.** A box or mask at least `ego_mask.max_fraction` inside
     the camera's `ego_mask` polygons (the rack, hood and antenna) never
     becomes or stays a track, and every mask loses those pixels.
   - **Back-fill.** YOLO's first box on a distant car is rarely its first
     appearance. After the forward pass, each track born after its window's
     first image is propagated backwards from its birth prompt for up to
     `backfill_frames` images (CTRL's 20 at 10 Hz; [CTRL](https://arxiv.org/abs/2304.12315),
     [OC-SORT](https://arxiv.org/abs/2203.14360) and
     [SAM4D](https://arxiv.org/abs/2506.21547) recover the frames before a
     first detection this way). One pass feeds the window's images to SAM2 in
     reverse order. It stops where the forward rules would end the track read
     backwards (`drop_lost_after_frames` empty masks, the ego vehicle) or where
     its mask's bbox fails the `untracked_area` test against the masks
     recorded there, since that object is already tracked.

   On CUDA OOM the camera is retried in windows of `sub_clip_frames`, halved
   down to `min_sub_clip_frames`; an OOM at that floor raises rather than
   publish a camera with gaps, and so does a failed late prompt or correction.
   SAM2 runs under `autocast_dtype` (bfloat16); the detector and DINOv2 stay
   float32. Retaining competing tracks can increase per-object memory attention,
   active mask memory and PNG writes until detector support or absent-mask
   checks end them. The existing OOM window retries still apply.

3. **Embeddings** (`models/embeddings.py`), from the masks pass 2 wrote: one
   appearance descriptor per masklet per image (every
   `embeddings.every_k_frames`-th image of a camera), for `tracking`'s ReID.
   The descriptor is Foreground Feature Averaging
   ([Kotar et al.](https://arxiv.org/abs/2311.00750); the code path of
   [NIDS-Net](https://github.com/YoungSean/NIDS-Net)): the mask's bounding-box
   crop resized to a 224 px square, DINOv2's patch tokens under the mask
   averaged, L2-normalised. The mask picks the tokens, so two objects sharing
   a bounding box differ, and the square resize keeps the ends of a wide car
   that a centre crop cuts off. Every image keeps its own descriptor, back-filled
   ones included: tracking compares two masklets by their closest pair, which
   separates nearby same-class objects far better than a mean per masklet
   (`docs/tracking_measurements.md`, "Appearance"). The descriptors are
   written before the tables that mark the chunk complete.

**One entry per image.** `frame_index` repeats an image once per sweep it is
nearest to (~4.9× on WATO, ~1.7× on nuScenes); both passes read it through
`wato_common.artifacts.frame_index.FrameIndex`, which hands out each image once
with every sweep it is paired with (depth anchors on all of them) and the pose
at the image's own time. Every frame-counted setting counts images.

Model loading and inference are fail-loud: a detector, SAM2, DINOv2 or DA-V2
failure stops the run instead of leaving a camera with fewer masklets or no
descriptors, both of which look like a finished chunk. SAM2 and the detector are
import-checked, and the SAM2, YOLO and DINOv2 weights load-checked on the CPU,
before each chunk's depth pass, so a missing checkpoint fails in seconds.

## Mask propagation backend

`segmentation.backend` picks how SAM2.1 Large (the same checkpoint either way)
carries a mask from image to image:

- `dam4sam` (default): the [DAM4SAM authors' multiobject
  tracker](src/wato_perception_2d/_vendor/d4sm/README.md), vendored at a
  pinned commit with its memory selection unchanged: three recent and three
  distractor slots besides the initial prompt, 16 object pointers. Our adapter
  shares one model and one image encoding per frame across objects; a late
  detection starts its own memory and a correction restarts only that
  object's (the authors have no correction API). Forward only; frames are
  read lazily on CPU. The adapter sets two memory parameters
  (`segmentation.dam4sam`), without editing the vendored files:
  - `update_delta`: a new memory every 2 images. The authors use 5 at 30 fps
    and 1 at 5 fps (one update per ~170-200 ms); their code's 5 would space
    WATO's recent memories 0.5 s apart.
  - `memory_min_predicted_iou`: [SAMURAI](https://arxiv.org/abs/2411.11922)'s
    memory-bank gate (0.5, from its code). The previous image is always
    attended to, but an image whose predicted IoU is at most the gate is
    never kept as an older recent memory, so a mask SAM2 itself doubts does
    not consolidate. Object pointers are not gated (nor in SAMURAI); the
    distractor memory keeps the authors' stricter IoU > 0.8. 0 disables.
- `sam2`: Meta's video predictor (`offload_video_to_cpu` applies to it only).

On the two nuScenes GT scenes the two score the same downstream; WATO is
unmeasured ("Mask propagation backend" in `docs/perception_2d_measurements.md`).
Manifests and resume caches record the backend, the SAM2 and DAM4SAM commits,
the adapter version and whether CUDA hole filling ran; each masklet records its
`tracker_backend`.

**CUDA hole filling.** Both backends fill background holes of ≤ 8 pixels in
SAM2's 256 × 256 mask with its `sam2._C` connected-components kernel, which
`docker/perception_2d.Dockerfile` compiles against the locked torch
(`SAM2_BUILD_ALLOW_ERRORS=0`; `SAM2_CUDA_ARCH_LIST` sets the GPU
architectures). SAM2 silently skips the fill when the kernel is missing or
built for another GPU, so on CUDA the pipeline checks it first (an 8-pixel hole
fills, a 9-pixel one stays) and stops with a rebuild message if not. On CPU
the fill doesn't run and the masks differ.

## Mask refinement (experimental, off)

`--refine-masks` (`segmentation.refinement.enabled`) seeds births and box-drift
corrections with image-predictor masks instead of boxes. Run it into a separate
artifact root: it needs `--force`, which replaces the chunk's outputs.

- At a keyframe needing a seed, SAM2's image predictor (the video model's
  weights) encodes the image once and predicts a mask from each box. It is
  used only if its predicted IoU ≥ `min_predicted_iou`, its threshold
  stability ≥ `min_stability`, it lies in its box and misses the ego mask;
  otherwise the box prompts as before, so weak distant detections survive.
- `interior_point` / `mask_input_refinement`: one more decoder pass with a
  click at the mask's most interior pixel and its low-resolution logits, kept
  only if predicted IoU rises, stability doesn't fall and the click stays
  inside.
- `correct_shape_disagreement`: also re-check confident supported tracks
  whose bbox still agrees, correcting only when the fresh mask's IoU with the
  propagated one is in [`correction_min_mask_iou`,
  `correction_below_mask_iou`) and it is at least as stable; a larger
  disagreement may be another object, and is only recorded.

The thresholds are SAM2's automatic-mask-generator defaults, not calibrated on
WATO, and no WATO gain has been measured. With stock SAM2, memory clearing on
correction stays off (the pinned SAM2 calls an undefined helper there).

## Mask diagnostics

`segmentation.mask_quality: true` writes `mask_quality/<cam_id>.json` per
camera once its tracking succeeds: per frame and object, raw / postprocessed /
saved pixel areas, threshold stability and, under DAM4SAM, its memory frames
and SAM2's own confidence (`predicted_iou` of the chosen mask,
`object_score_logit`); per box the birth gate tested, its measure and
verdict; per back-filled track, the masks added, their predicted IoU and
why it stopped; per
refinement prompt, its scores and whether it was applied (`mask_quality.py`).
Off by default: it costs a GPU sync per frame and 10-20 MB of JSON per WATO
chunk. Stability (`area(logit > +offset) / area(logit > −offset)`), predicted
IoU and object score are model confidence, not accuracy; nothing downstream
reads them. Published trackers end tracks and re-prompt on predicted IoU and
object score (Seg2Track-SAM2, SAM2MOT); whether they separate drifted masks
from real cars on WATO is what these fields are for.

`scripts/diagnose_masks.py` tracks one camera's frame range (fixed taxonomy +
YOLO, no depth or DINOv2) into an empty artifact root, with the report and a
`summary.json`, for comparing backends without a full run:

```bash
python3 /ws/src/perception_2d/scripts/diagnose_masks.py \
  --config /ws/src/perception_2d/config/perception_2d.wato.yaml \
  --bag <bag_id> --chunk 0000 --cam PANO_NN --frames 120 \
  --backend dam4sam --output /data/artifacts/mask_diagnostic_dam4sam
```

## Detector backends

`detection.backend` picks the box source; both return the same boxes:

- **`yolo`** (default): Ultralytics YOLO26-x, COCO weights, closed-set.
  `detection.yolo.class_map` maps COCO onto the taxonomy; a taxonomy class
  nothing maps to is never detected (logged at startup). It needs
  `discovery.backend: fixed`. AGPL-3.0.
- **`grounding_dino`**: open-vocabulary, text-prompted by the taxonomy or by
  Florence-2 discovery (`discovery.backend: florence2`). Its scores depend on
  which other words share the prompt.

The taxonomy is the six classes COCO covers: car, truck, bus, motorcycle,
bicycle, pedestrian. Traffic cones and barriers are left out: COCO has neither,
GroundingDINO's hits for them were mostly hydrants, poles, walls and the ego
rack, and a class listed but never detected would be labelled absent from every
image.

`models/yolo_detector.py` guards three silent failures: Ultralytics reads numpy
as BGR (frames are RGB), weights load by pinned path (a bare name downloads
whatever upstream ships), and `YOLO_OFFLINE` / `YOLO_CONFIG_DIR` are set before
import. The pipeline refuses a `class_map` target that nearly matches a
taxonomy class (`pedestrain`, `Motorcycle` vs `motorcycle`).

## Outputs

In `data/artifacts/raw/<bag_id>/chunks/<chunk_id>/`:

| Artifact | Owner |
|---|---|
| `tracklets_2d.parquet`, `detections_2d.parquet`, `masks_2d/<masklet_id>/<camera_seq:06d>.png` | `wato_common.artifacts.masklets` (consumers use `MaskletIndex`) |
| `masklet_embeddings.npz` | `wato_common.artifacts.masklets` (`read_embeddings`): one unit-norm float16 descriptor per (masklet, embedded image), with the model and pooling that made it; absent with `embeddings.every_k_frames: 0` |
| `depth_2d/<cam_id>/<camera_seq:06d>.npz` | `wato_common.artifacts.depth`: a missing file means no usable depth; a 0 pixel means no measurement |
| `manifest_perception_2d.json` | Provenance incl. pinned model revisions, hashed inputs |
| `mask_quality/<cam_id>.json` | Perception-local mask diagnostics, only with `segmentation.mask_quality` (`mask_quality.py`) |

Masklet rows declare `mask_semantics: independent`. Readers reject old rows
without this marker because saved owned masks cannot recover their losing
pixel claims. Re-run perception with `--force`, then semantic lifting and
proposal generation, to regenerate those claims. Per-camera resume caches
also record this policy and reject caches made with owned masks; `--force`
clears those caches before processing.

Outputs from before 2026-10-15 (per-row processing) aren't comparable, and
depth from before 2026-10-11 is rejected by the reader; rerun with `--force`.

## Models

Every model is pinned in
[`model_registry.py`](src/wato_perception_2d/model_registry.py), which both
`fetch_models.py` and the runtime loaders read. HF loaders pass
`revision=hf_revision(repo_id)` (a SHA fetch writes no `refs/main`, so `main`
can't resolve offline), and DINOv2 loads `facebookresearch/dinov2:<sha>`
(torch.hub caches by ref). A model ID not in the registry fails at load time:
register it with a SHA first. To upgrade one, change its revision, re-run
`fetch_models.py` and the affected stages; manifests record the revision.

## Ego mask

`ego_mask` (per camera, pixel polygons in the rectified image) is the only
per-vehicle part of the config; `perception_2d.yaml` has none, and each rig
gets its own file (`perception_2d.wato.yaml`). `ego-mask` drafts and checks it
from a bag's perception_2d output, since the detector boxes the vehicle's body
and SAM2 holds it all chunk: a mask at the image border that doesn't move
while the vehicle drives. Into `raw/<bag_id>/ego_mask/` (or `--out`) it writes,
per camera, the median of images taken metres apart (the scene blurs, the body
stays sharp), an overlay of those body masks (red) and the config's polygons
(green), and `ego_mask.yaml`, a draft block from the body masks.

- **New vehicle:** run perception_2d on a driving bag without an ego mask, then
  `ego-mask`; check the overlays, add by hand what no masklet covers (the WATO
  antenna), and put the block in the rig's config.
- **Check:** after a run with the polygons, `ego-mask --config <rig yaml>`
  reports body masklets left outside them (a camera moved, a rack added).
  None means the polygons still hold. The polygons are tied to the image size
  (a mismatch raises) and to the rectification; nothing else detects drift.

## Comparing two runs

`compare` reviews what a change to the tracking rules removed, against a run
of the same bag from before the change. It is one-sided: for a change that
adds masks (back-fill), run it again with the roles swapped, the new run as
baseline, to review what was added. Both runs box the same images with the
same detector, so an object the change lost is a baseline mask that no current
mask matches (pixel IoU < 0.5), checked on every `--every`-th image (5).

```bash
./watod run perception_2d compare --bag <bag_id> --baseline /data/artifacts/<root> \
    [--chunk <chunk_id> ...] [--config <the current run's yaml>]
```

The baseline is another artifact root: the old code run with
`ARTIFACT_ROOT_URI` pointing at it, its chunks linking the current root's
ingest and lidar_preprocessing outputs (depth can stay off). Into
`raw/<bag_id>/compare/` (or `--out`) it writes:

- `review.csv`: the baseline masklets with unmatched masks, largest loss first;
- `sheets/`: pictures of them (baseline mask red, the current mask over it
  outlined green);
- `masklets.csv` and `summary.txt`.

Give each review row a verdict: `real` (an object the current run lost),
`junk` (not an object: ego body, sky, a driver), `covered` (still masked, by
another masklet or a trimmed mask) or `unsure`; masklets inside `--config`'s
ego mask are pre-marked junk. Run it again: the verdicts are kept, and the
summary gives per class the share of the baseline's real-object masks lost,
with unsure and unreviewed rows counted as real for an upper bound. It can't
see an object neither run masks.

## Depth viewer

`viz` opens a matplotlib window per chunk: a frame slider, an RGB ↔ depth
opacity slider, a draggable split curtain, camera switching; keys `←/→` frame,
`↑/↓` opacity, `space` split, `c` colormap, `q` quit. Depth is normalised per
frame (2nd–98th percentile); the colorbar reads metres and the title shows fit
status, inliers and RMSE. Needs DISPLAY / WSLg in the container, or run it on
the host:
`PYTHONPATH=src/common/src:src/perception_2d/src python -m wato_perception_2d viz --bag <bag_id>`.

## Known limitations

- **Ego vehicle.** The WATO ego mask is for one camera mounting, traced on
  `sensors_2026_07_07-23_01_24_2`; a remounted camera needs it redrawn (see
  "Ego mask"). Under the default `perception_2d.yaml` (no ego mask) the rack is
  tracked as a truck.
- An object occluded for longer than `drop_lost_after_frames` comes back as a
  new masklet; only `tracking` can rejoin it, by its 3D boxes or, with
  tracking's `appearance` on, by the descriptors.
- **Birth gate.** `untracked_area` also refuses a real car under a large mask
  of junk (sky, road) until detector support ends that mask (about a second);
  back-fill then recovers the car's earlier images. No published 2D rule
  repairs a mask SAM2 confidently bleeds over two cars; the gate only stops it
  from also spawning a duplicate.
- **Back-fill** can follow an object behind an occluder onto its neighbour
  with SAM2 still confident (predicted IoU 0.7-0.8), so neither stop rule nor
  a published quality gate ends it. On nuScenes it adds ~2% box recall for
  1-2 points of precision, and on scene-0061 it gives one parked car extra
  points (+0.4 m wide, +1.1 m long). Measurements:
  `docs/perception_2d_measurements.md`.

Measurements behind these defaults:
[docs/perception_2d_measurements.md](../../docs/perception_2d_measurements.md).

## Camera calibration source

`calibration_source: recorded` is the default. `validated` requests the
camera-calibration stage's accepted result and fails on missing, stale,
unqualified or failed-camera evidence. Keep this choice consistent across
camera-dependent stages. Output/cache identities reject mixing calibrations;
rerun depth and downstream processing with `--force` after a change. See
[the calibration module](../camera_calibration/README.md).
