# wato_world — Claude project guide

## Prime directive: mirror `wato_monorepo`

Derive everything here — architecture, infrastructure, code style, naming,
shell scripts, Dockerfiles, CI workflows, directory layout — from
`wato_monorepo` (a sibling clone, e.g. `../wato_monorepo`). Before writing a
shell script, Dockerfile, workflow or helper, grep the monorepo for the
equivalent and copy its shape, adapting only where offline batch genuinely
differs. Invented infrastructure has had to be ripped out and re-aligned with
the monorepo before.

Reference points: `watod` (entrypoint shape, flag parsing, `ACTIVE_MODULES`
with `:dev`), `watod_scripts/watod-compose.sh` (pull bases, then build),
`docker/base/inject_*.Dockerfile` (`ARG GENERIC_IMAGE` injection),
`docker/template.Dockerfile` (`source` / `dependencies` / `deploy` / `develop`
stages), `.github/workflows/build_base_images.yml`, `.pre-commit-config.yaml`.

Deliberate divergences: no runtime ROS messaging (no Zenoh, no `network_mode:
host`, no `package.xml` / colcon) — components talk through artifacts on disk;
no `infrastructure`, carla or simulation profiles; config is per component at
`src/<component>/config/<component>.yaml`.

## What this repo is

Offline batch 3D auto-labeling for WATonomous: eleven components, one Docker
image each, communicating only through files under `data/artifacts/`
(Parquet / JSON / PNG / NPZ). No runtime messaging, no database.

```
ingest → lidar_preprocessing → camera_calibration → perception_2d → semantic_lifting →
proposal_generation → tracking → label_refinement / open_vocab_discovery → student_training
         lidar_preprocessing → detection_3d ↗ (BEVFusion boxes into proposal_generation)
```

ingest, lidar_preprocessing, perception_2d and semantic_lifting are implemented
end to end; detection_3d runs the car's BEVFusion (camera + LiDAR, int8,
TensorRT) on every frame, with the inputs offline allows (nuScenes' training
frame, earlier sweeps, images at nuScenes geometry); proposal_generation has
its sources (Step F clusters, lifted instances, detection_3d's boxes as
`fusion_detector`, fusion; no LiDAR-only detector); tracking has a v0 (tracks
per chunk with SimpleTrack's GIoU cost, never-terminating tracks and
CenterPoint's circle NMS on vehicles, re-linked after the pass by
Offline-Poly's STWO, then linked across chunks; a parked / moving state per
track; appearance ReID from perception_2d's masklet descriptors built in and
off by default; `wato_common.tracking` keeps Chen's cost as the default
for Step F); label_refinement has a v0 (one box per parked vehicle, one size and a
smoothed path per moving track, then SLF's box shape and heading fit per
track to its masks and points, PyTorch on the GPU); the rest are stubs raising
NotImplementedError. Each implemented component's README is its
reference.
Measurements behind defaults are in `docs/*_measurements.md`; which recordings
are usable is `docs/drive_bag_survey.md`. Scoring against ground truth: import
it into `gt_boxes` (`src/ingest/scripts/`: nuScenes annotations from a converted
bag, or boxes drawn in SUSTechPOINTS), then run a stage's `scripts/eval_*.py`,
which share `wato_common.eval`; the nuScenes scene is
`NuScenes_v1_0_mini_scene_0061_0_mcap`, and the same scene ingested as two
chunks, `NuScenes_v1_0_mini_scene_0061_2chunk`, scores a chunk seam
(`docs/tracking_measurements.md`).

perception_2d's mask tracker is `segmentation.backend`: `dam4sam` (default;
the authors' source vendored at a pinned commit in `_vendor/d4sm/`, whose two
copied files must stay byte-identical) or `sam2`, both on the same SAM2.1
weights; they score the same on the nuScenes GT scenes, and WATO is
unmeasured. The DAM4SAM adapter sets two memory parameters from its context
subclass (`segmentation.dam4sam`: update spacing 2 images, SAMURAI's RAM gate),
never by editing the vendored files. A box starts a track only past
`tracking.birth_gate` (SAM2MOT's untracked area against tracked masks, or SAM 3's
mask IoU), and each later birth is back-filled backwards from its prompt
(`tracking.backfill_frames`). A third pass writes one DINOv2 appearance
descriptor per masklet per image (`masklet_embeddings.npz`, mask-pooled patch
tokens) from the saved masks; `./watod run perception_2d embed` redoes only
that pass. CUDA runs need SAM2's compiled `sam2._C` hole-filling kernel,
checked before the depth pass. Mask refinement (`--refine-masks`) and mask
diagnostics (`segmentation.mask_quality`) are off by default. Details:
`src/perception_2d/README.md`.

## Conventions

- Layout: `src/<component>/src/wato_<component>/`, tests in
  `src/<component>/tests/`, config in `src/<component>/config/`; shared code in
  `src/common/src/wato_common/`.
- **One module per cross-component artifact.** Anything one component writes
  and another reads is encoded, written and read by its module in
  `wato_common/artifacts/` (`calibration`, `frame_index`, `lidar_sweeps`,
  `motion_proposals`, `ground`, `masklets`, `depth`, `lifted_labels`, `lifted_candidates`,
  `detections_3d`, `proposals`, `tracks`, `refined_labels`; `gt_boxes` is imported ground
  truth that only evaluation reads),
  round-trip tested in
  `src/common/tests/test_artifacts_*.py`. Producers call the write half,
  consumers the read half; nobody else `np.load`s or re-derives the encoding.
  Consumer images install only `src/common` + their own package, so both halves
  live in common (the monorepo's `*_msgs` packages play this role). Reads raise
  `ArtifactFormatError` instead of defaulting a missing value. A new
  cross-component artifact = a new module + its round-trip test.
- **One artifact layout**: `raw/<bag_id>/...` and
  `raw/<bag_id>/chunks/<chunk_id>/...` from `wato_common.artifact_store`, for
  every component. No version directories; provenance lives in manifests.
  Per-sweep files are named `<sweep_id:06d>`; `sweep_id` is int64 everywhere.
- Docker: `docker/<component>.Dockerfile`, bases
  `docker/base/inject_{cpu,cuda}_base.Dockerfile`; compose
  `modules/docker-compose.yaml` (deploy) + `.dev.yaml` + `.gpu.yaml`.
- **Docs move with code.** A change to observable behaviour, an artifact
  schema, `frame_index` fields, a flag or a cross-component assumption is
  complete only when the component README, this file, config comments and
  docstrings match it. Comments state the constraint and its reason; history
  belongs in commit messages, measurements in `docs/`.

## watod

```bash
./watod build                 # pull bases from ghcr, build ACTIVE_MODULES (watod-config.sh)
./watod up                    # docker compose up -d
./watod -t ingest_dev         # shell into a running dev container
./watod run <component> --bag <bag> [flags]   # also: validate-bag, trim-bag, debug-bag, reduce, viz, ego-mask, compare, embed, pick
./watod pipeline --bag <bag> [--profile wato] [--force <stage>,...] [--from|--to <stage>] [--dry-run]
./watod test <component>      # src/common/tests, then the component's, in its dev image
./watod down all
./watod build-base            # only when ghcr lacks the bases (first-time setup)
```

`ACTIVE_MODULES` matches the monorepo: `"ingest:dev"`, `"all"`, `"all:dev"`, or
a list; `:dev` = develop target with source bind-mounts.
`watod pipeline` (`watod_scripts/watod-pipeline.sh`) runs ingest through
label_refinement through `watod-run.sh`. Every stage skips work whose outputs
exist (ingest: each chunk's `manifest.json`; lidar_preprocessing's bag-level
steps: outputs newer than their inputs), so `--force <stage>` re-runs that
stage and every later one. `--profile <name>` takes each stage's
`<stage>.<name>.yaml`, falling back to the longest existing prefix (`wato_6cam`
→ `wato`) and then `<stage>.yaml`. A stage not active as `:dev` runs its image
as last built. Run tests with
`./watod test`: the images have pyarrow, ROS, numba and pypatchworkpp. On a host
with the Python deps,
`PYTHONPATH=src/common/src:src/ingest/src python3 -m pytest -p no:anyio src/ingest/tests`
runs ingest's suite without ROS (`-p no:anyio`: the host's anyio plugin
doesn't load; the trim-bag end-to-end test skips).

## Cross-component contracts

- **`frame_index` has one row per (LiDAR sweep, camera)**, so an image repeats
  once per sweep it is nearest to (~4.9× on the 3-LiDAR WATO rig). Consumers
  read it only through `wato_common.artifacts.frame_index.FrameIndex`: each
  image once (`by_camera()`, `for_sweep()`), with every sweep it is paired with
  (`sweep_ids`, perception_2d's depth anchors) and its pose at the image's own
  time (`world_T_ego`, None in an untrusted stretch). Feeding rows straight to
  SAM2 got the process SIGKILLed on a 30 s WATO chunk. lidar_preprocessing
  doesn't read it: it checks each sweep's pose with `PoseLookup` itself.
- **Every pose is looked up at its datum's own time**: a sweep's, a deskewed
  point's, a camera frame's, all interpolated from `poses.parquet` through
  `wato_common.pose_lookup.PoseLookup`, which honours ingest's per-stretch
  verdicts. frame_index stores no pose; projecting into an image uses
  `FrameIndex`'s `CameraFrame.world_T_ego`, looked up at the image's time, not
  the sweep's (up to 44 ms apart on WATO, 120 ms on nuScenes).
- **`sweep_id` is one counter per chunk across all LiDARs**, so per-sweep files
  and joins key on `sweep_id` alone. Chunks ingested before that restart it per
  LiDAR (the three WATO scanners overwrite each other's world files) and must be
  re-ingested.
- **Neighbouring chunks share 6 s, and every id restarts per chunk.** Ingest
  decodes each chunk over `t_overlap_*`, its nominal window ±
  `chunk_overlap_seconds`, and nominal windows start
  `chunk_seconds - chunk_overlap_seconds` apart, so neighbours share
  3 × `chunk_overlap_seconds`. `sweep_id`, `camera_seq`, masklet,
  lifted-instance and `track_hint_id` ids are chunk-local: match a sweep or
  image across chunks by `(lidar_id, reference_timestamp_ns)` /
  `(cam_id, camera_timestamp_ns)` (a sweep decoded into both chunks has
  identical world points in both). Each instant of the span belongs to one
  chunk, split at the midpoint (`wato_common.chunk_spans`): tracking marks its
  rows by it (`owned`), and every `eval_*.py` scores an instant once, in its
  owner. tracking links tracks across the span: `src/tracking/README.md`.
- **lidar_preprocessing has two dynamic artifacts.** `*_dynamic_mask.npy` /
  `dynamic_map.npz` is Step B's *precision* verdict after its point filter
  (`~mask` is trusted static: perception_2d's depth anchors), read through
  `wato_common.artifacts.lidar_sweeps`, which checks it aligns with the world
  sweep. `*_motion_proposals.npz` / `motion_clusters.parquet` (Step F) is the
  *recall* artifact (per-point `source_bits` + HDBSCAN clusters with soft motion
  features; false positives by design), owned by
  `wato_common.artifacts.motion_proposals`. Step E (`iwu/`, UniLiPs IWU →
  `global_iwu.npz`) evicts floaters from the bag map and feeds Step F;
  `--two-pass` is a one-time log-odds prior, not IWU. Step F's box / Kalman /
  association code is `wato_common/tracking/`, for the `tracking` component to
  reuse.
- **Lifted instances are chunk-local merged masklets.** `tracking` (the owner
  of cross-camera identity) runs after semantic_lifting, so lifting merges one
  object's per-camera masklets itself from the LiDAR points they both claim;
  `lifted_labels` name the instance by its smallest masklet_id and
  `lifted_instances.parquet` lists its masklets and cameras.
  Perception saves independent overlapping masks (`mask_semantics: independent`).
  Lifting preserves all visible point-instance claims in `lifted_candidates`;
  `lifted_labels` is a provisional unique view. Proposal generation refines
  candidates before exclusive ownership, and only surviving camera seeds
  class motion proposals. When changing ownership or regenerating old owned
  masks, read `docs/research/deferred_3d_mask_ownership.md` for ordering and reruns.
- **Proposals are per frame and per source.** One `ProposalRow` per object per
  frame_sync frame, keyed on the frame's reference `sweep_id` (the canonical
  LiDAR's; `frame_id` is None for a sweep matched to none). Every source's rows
  (`lidar_mos`, `lifted`, `fusion_detector`; a LiDAR-only `lidar_detector`
  later) stay beside the
  `fused` rows that list them in `member_proposal_ids`, so each source's
  reliability can be measured; tracking reads the fused rows. SLF is not a
  proposal source: it fits one shape per track in label_refinement. `cls` is
  perception_2d's class (`car`, `truck`, ...), else the detector's mapped onto
  that taxonomy, None when neither supports the box. No artifact records lidar_preprocessing's canonical LiDAR,
  so proposal_generation's config restates it (`frames.canonical_lidar`,
  `lidar_cc` in `.wato.yaml`) and `ProcessedSweeps.frames` raises
  `CanonicalLidarError` on a mismatch.
  A matched `fusion_detector` contributes provenance while preserving the
  motion/lifted geometry, class, score and support. Unmatched detections add
  objects only at score ≥ 0.6 with ≥ 5 non-ground points. Tracking prioritizes
  independent boxes during duplicate suppression and association; matched
  detector scores never promote their strength. Detector + lift is one
  camera evidence family. The WATO profile enables this supplemental
  policy; older geometry-replacement measurements describe a different policy.
  Lifted support has opt-in full non-ground scan-cluster matching. When changing
  refinement or point-count evidence, read `docs/research/sal_scan_cluster_refinement.md`
  for SAL fidelity choices; tracking counts retained camera seeds separately
  from recovered geometry.
- **detection_3d keeps every box, raw and in the world.** BEVFusion's 200
  boxes per frame, whatever the score or class, keyed on the frame's
  reference `sweep_id`, with `world_T_model` per frame: thresholds, the
  class map and the car's 2.5 m ego zone are proposal_generation's
  (`fusion_detector`). Vendor z is the box's centre (MIT's loader read
  nuScenes centres as bottoms), not the bottom its decoder implies; heading
  is directed. Frames are anchored on the canonical LiDAR; a frame without
  every configured camera within 200 ms is skipped and recorded as such.
  WATO uses all eight top cameras and only `lidar_cc`, with a native eight-view
  ONNX export and camera-count-aware CUDA binding. When changing sensors,
  exports or detector influence, read `docs/sensor_setup.md`; cached inputs
  must match and sensor changes require an ingest-through-refinement rerun.
- **Tracks are bag-level.** tracking tracks each chunk's fused rows, links
  each chunk's tracks to the next chunk's that agree in their shared span, and
  writes bag-level `tracks.parquet` + `track_summaries.parquet`. A linked
  track (`track_id` `<chunk_id>-<n>`, after its first chunk) has rows from each
  of its chunks, so a shared-span instant can have two rows; `owned` marks the
  one its chunk labels, consumers read only owned rows, and
  `tracks.per_instant` keeps one row per instant. A track's `motion_state`
  (moving / parked / unknown) is judged on the whole linked track
  (`src/tracking/README.md`), not by Step F's per-cluster `motion_score`;
  label_refinement refines it whole.
- **Appearance is per masklet per image, compared by closest pair.**
  perception_2d writes `masklet_embeddings.npz` (unit-norm rows, with the
  model and pooling that made them), read only through
  `wato_common.artifacts.masklets.read_embeddings`; rows of different
  model or pooling don't compare. tracking's `appearance` (off by default)
  uses them in the re-link under a threshold calibrated on the bag's own
  masklet pairs. When changing the descriptor, its use or the default, read
  `src/tracking/README.md` ("Appearance") and the measurement behind it in
  `docs/tracking_measurements.md`; a descriptor change needs `embed --force`
  and a tracking rerun.
- **Sensor physics lives in `sensor_model.py`** (profiles `vlp32c`, `vlp16`,
  `hdl32e`), never in YAML.

## Camera calibration

`camera_calibration` runs after LiDAR preprocessing, before detection/depth.
It uses a pinned Koide NID objective in a headless Ceres runner, fixed camera
intrinsics/LiDAR mounts/poses, and independent static correspondences for
validation. Shipped profiles stay disabled and unqualified: current WATO
fits can be unstable. `./watod run camera_calibration pick` writes a browser
picker for those correspondences on the report's held-out views; pixel and
3D point are picked separately and LiDAR is never drawn on the image, so picks
stay independent of the calibration they judge. When changing calibration selection, validation, cache
identity or rollout, read `src/camera_calibration/README.md` and
`docs/camera_calibration_measurements.md`.

Ingest's `calibration.json` stays baseline provenance. Candidate/effective/report
artifacts live beside it under `camera_calibration/`, read/written only through
`wato_common.artifacts.calibration_refinement`. Camera consumers explicitly choose
`calibration_source: recorded|validated`; validated reads reject stale,
unqualified or failed-camera results. Calibration identities bind resumable
caches and propagate through proposal/tracking/refinement manifests. Rerun all
camera-dependent outputs with `--force` after changing their calibration.

## Ingest

Details: `src/ingest/README.md`; measurements: `docs/ingest_measurements.md`.

- **Dataset-agnostic: a new recording needs only a new YAML.** Nothing in
  `wato_ingest` names a dataset, vehicle or topic (the `wato_` prefix is the
  repo namespace). A recording's quirk goes in its profile YAML or is rejected
  with a named error. Ingest validates topic types up front, detects the
  rosbag2 storage plugin, and stops rather than write artifacts that look
  complete but aren't.
- **Profiles.** `ingest.yaml` (nuScenes); `ingest.wato.yaml` (8 rectified pano
  cameras, only `lidar_cc`, eidos `liso/odometry`, `ego_frame:
  base_footprint`); `ingest.wato_6cam.yaml` = `ingest.wato.yaml` without
  `PANO_EE` / `PANO_WW` (`ring_road_corrected`, May 30 tests 2–3);
  `ingest.wato_no_lidar_nw.yaml` is a compatibility alias of `ingest.wato.yaml`; a test
  fails if any other value drifts. lidar_preprocessing's `.wato.yaml` pairs
  with them (`per_lidar`: `lidar_cc` vlp32c;
  `frame_sync.canonical_lidar: lidar_cc`; `header_stamp_at: sweep_end`, from the
  car's `timestamp_first_packet: false`, since WATO Velodyne `time` fields are
  all zero and deskew synthesizes times from azimuth). So does
  `perception_2d.wato.yaml` = `perception_2d.yaml` + the rig's per-camera
  `ego_mask` polygons (roof rack, hood, antenna; a test fails on any other
  drift): without it the rack is tracked as a truck. `./watod run perception_2d
  ego-mask` drafts a new rig's polygons from a run without them and, after a
  run with them, reports body masklets they miss.
- **WATO LiDAR: ingest only `/lidar_cc/velodyne_points`**:
  classify's ray traversal assumes one sensor origin per sweep, and merged
  clouds leak static structure into `dynamic_map.npz`.
- **Pose source: dense and smooth.** `pose_requirements` aborts a chunk whose
  pose span is under `min_dense_fraction` covered by samples ≤ `max_bracket_ms`
  apart, and marks gaps and jumps in `poses.parquet` (`interval_drop_reason`);
  nothing inside them gets a valid pose. Passing is necessary, not sufficient.
  Use eidos `liso/odometry` (per-scan, stamped at the scan; only as dense as the
  recorded `lidar_cc` scans — `recording_20261004_185825_1` aborts). Not
  `slam/odometry` (newest-keyframe pose: one per 5 m on eidos main; on
  `ring_road_corrected`, LISO poses stamped 244–450 ms late, uncorrectable by a
  shift). Not `/novatel/oem7/odom` (~2° attitude bias vs the LiDAR, mount vs INS
  undetermined; no profile ships). Never `/tf` (eidos_transform stamps it with
  wall-clock time). A bag without eidos topics: run
  `wato_monorepo/tools/eidos_offline_to_bag.sh` and ingest the
  `<bag>_with_eidos.mcap` it writes.
- **Calibration** comes from the bag (each camera's `info` topic, the first
  PointCloud2 per LiDAR, a BFS over the TF topic) or `--calibration <file>`;
  `require_complete()` aborts on a sensor without intrinsics or extrinsics.
  WATO records the rectify node's output (JPEG `image_rect_compressed` on most
  recordings; raw bgr8 `image_rect` on Feb 15 / Mar 9 — repoint the image
  topics), described by `CameraInfo.P[:3,:3]` with no distortion, so those
  cameras are `rectified: true` and ingest writes `K = P[:3,:3]`,
  `distortion = []` (raw CameraInfo kept). Projection and inverse rays use
  `CameraCalibration.model`, including full K and the stored image's lens
  model. When changing projection, depth anchors, SLF rays or image geometry,
  follow `docs/camera_geometry_contract.md` for supported models and image
  normalization constraints. Accepted SLF boxes use the fitter's evaluated
  positions and headings for every frame without another smoothing pass.
- **Before ingest, `validate-bag` / `trim-bag`.** Judged on log time. trim-bag
  uses the pinned `mcap` CLI (rosbag2's writer would empty custom-type schemas)
  and writes to `data/bags/trimmed/`; with `data/bags/debug/` the only
  read-write mounts under `/data/bags`. One-shot eidos topics must be passed as
  `--static-topic`, or there is no common window.
- **`debug-bag` (ingest) merges the bag's boxes into a copy of the source bag**
  as `visualization_msgs/MarkerArray` topics (`/wato_world/proposals/<provenance>`,
  `/wato_world/tracks`, `/wato_world/refined_labels`), in the ego frame at
  each frame's reference time (the bag has no usable TF for the pipeline's
  world frame); each instant from the chunk that owns it, or one `--chunk`
  over its decoded window. `--masks` / `--depth` add perception_2d's masklets
  and depth drawn per camera image (`/wato_world/{masks,depth}/<cam>`, JPEG
  `CompressedImage`), stamped with the source image's header stamp, log time
  and `frame_id`. `src/ingest/README.md`.

## CI / ghcr

Three workflows: `build_base_images.yml` (pushes `base:cpu-ubuntu24.04` and
`base:cuda12.8.1-cudnn-runtime-ubuntu24.04` to ghcr, on pushes to main touching
`docker/base/**` or `workflow_dispatch`), `build_and_test.yml` (builds each
component against the published bases and runs pytest, on PRs and main; fails
with `not found` / `403` until the bases exist), `pre-commit.yml`.

First-time setup: Repo Settings → Actions → Workflow permissions = "Read and
write"; the workflows must be on `main`; run build-base-images manually and wait
for both cpu and cuda jobs to log `pushing manifest for
ghcr.io/watonomous/wato_world/base:...`. The `org.opencontainers.image.source`
label links the package to the repo; on 403s, republish the bases, or make the
package public / grant the repo read access.

- `${{ github.repository }}` keeps case (`WATonomous/wato_world`) and ghcr tags
  must be lowercase: keep the workflow's `tr '[:upper:]' '[:lower:]'` step.
- `build_and_test.yml` needs `docker/login-action` before build-push even for
  public packages (buildx tries an anonymous token first, which ghcr 403s for
  org packages).
- 403 = unauthenticated or no such package; `not found` while authenticated =
  never published.
- Ubuntu 24.04 ships Python 3.12; the bases use unpinned `python3`.
- `watod build` pulls bases and never builds them, like the monorepo;
  `watod build-base` (`--build-arg GENERIC_IMAGE=...`) is only a fallback.
- Check: `docker pull ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04`.
  Private packages: `echo $GH_PAT | docker login ghcr.io -u <user>
  --password-stdin` (PAT scope `read:packages`).

## Reproducibility — pinning and provenance

The output is training data: a rebuild that silently resolves a different
dependency set, model revision or base image is a different labeler, and its
labels aren't comparable.

| Surface | Where | Mechanism |
|---|---|---|
| Python deps | `docker/requirements/{ingest,lidar_preprocessing,perception_2d,semantic_lifting,detection_3d,proposal_generation,tracking,label_refinement}.txt` | exact `==` for the full transitive closure, installed `--no-deps` |
| `sam2` | `docker/perception_2d.Dockerfile` | `ARG SAM2_COMMIT` (no PyPI release); manifests read the installed package's commit back, so there is no second pin |
| Model weights | `src/perception_2d/src/wato_perception_2d/model_registry.py` | HF `revision=`, torch.hub `repo:<sha>`, raw `.pt` from an HF commit loaded by path |
| Component bases | `docker/{ingest,lidar_preprocessing,perception_2d,semantic_lifting,detection_3d,proposal_generation,tracking,label_refinement}.Dockerfile` | `base:<tag>@sha256:...` |
| BEVFusion runtime | `docker/detection_3d.Dockerfile` | `ARG WATO_CUDA_BEVFUSION_COMMIT`, `PYBIND11_COMMIT`, `TENSORRT_VERSION` + exact apt versions; `CUDASM` |
| BEVFusion model | `src/detection_3d/config/detection_3d*.yaml` | `model.sha256` per ONNX file; engines rebuilt per (files, TensorRT, GPU) |
| Upstream bases | `docker/base/inject_*.Dockerfile` | `ubuntu` / `nvcr.io` by digest, `ARG UV_VERSION` |
| `mcap` CLI | `docker/ingest.Dockerfile` | `ARG MCAP_CLI_VERSION` + `MCAP_CLI_SHA256` |

Unpinned: the other two Dockerfiles, both stubs (mutable `base:<tag>`, no
lockfile, no `WATO_BASE_IMAGE`, no manifest) — pin each when it stops being
one; and ingest's ROS 2 Jazzy apt packages (no apt lockfile;
noted in `docker/requirements/ingest.txt`).

**Lockfiles are generated.** Change the intent in the component Dockerfile,
build its deps stage, re-freeze, and commit both together:

```bash
docker run --rm --entrypoint uv \
    ghcr.io/watonomous/wato_world/<component>:deps_<tag> pip freeze --system \
    | grep -v '^Using Python'
```

Verify a lock still reproduces by diffing that output, sorted, against
`sort docker/requirements/<c>.txt | grep -v '^#'`. perception_2d and
label_refinement (torch) need `--index-strategy unsafe-first-match`:
`download.pytorch.org/whl/cu128` mirrors common packages at its own versions,
and uv's default then refuses PyPI. Adding a dependency, resolve it with the
current lock as constraints (`-c`) so nothing already pinned moves.

**Provenance** (`wato_common/`): `provenance.py` reads `WATO_GIT_COMMIT`,
`WATO_GIT_DIRTY`, `WATO_BUILD_TIME`, `WATO_BASE_IMAGE` and the hash of
`/opt/watonomous/requirements.lock.txt`; `manifest.py` writes one manifest per
(bag, chunk) per component with content-hashed inputs (> 64 MB: size + mtime),
so a changed input marks the outputs stale. ingest writes `manifest.json`;
lidar_preprocessing, perception_2d, semantic_lifting, detection_3d,
proposal_generation, tracking and label_refinement write
`manifest_<component>.json` (semantic_lifting, detection_3d and
proposal_generation hash their upstreams' manifests in place of their
per-sweep files; detection_3d also records its model files' hashes, TensorRT,
GPU and fork commit; tracking's and label_refinement's outputs are
bag-level, so each chunk's manifest names the same files).

- `WATO_GIT_COMMIT` is baked in at build time (`watod-setup-env.sh` →
  `modules/.env` → build arg): deploy images have no `.git`.
- `ARG BASE_IMAGE` is global; re-declare it inside the `dependencies` stage or
  `ENV WATO_BASE_IMAGE=${BASE_IMAGE}` expands to empty.
- Build `_source` profiles before the runtime profile: one `docker compose
  build` runs them in parallel and ships stale code. `watod build` orders them.

Known drift frozen into the locks, to fix as separate testable changes:
perception_2d declares `numpy<2` but ships 2.5.3; perception_2d has both
`opencv-python` and `opencv-python-headless`; `depth-anything-v2` pulls in
gradio, fastapi, uvicorn, starlette.

## Pre-commit gotchas

- `PermissionError: '.codex'`: a deleted file still in the index — `git rm .codex`.
- F821 `Undefined name 'reader'` in `rosbag_reader.py` comes from a `del reader`
  in the contextmanager's `finally`; leave the cleanup to refcounting.

## Removed on purpose

Postgres / any database (`psycopg`, `sqlalchemy`, `PG_DSN`, `PG_PORT`,
`docker-compose.infra.yaml` references are stale; the table shapes are
`wato_common.schemas`), `watod bag` / `watod-bag.sh`,
`watod_completion.bash` (port it from the monorepo if wanted),
`watod-config.local.sh.example`. MF-MOS, with `--seg mos|union`, `union/` and
torch/CUDA in the lidar_preprocessing image: Step B is the AW classifier, CPU
only, and bit 3 of Step F's `source_bits` stays unused.
The `traffic_cone` / `barrier` classes and SAM 3 (as segmenter or detector):
no detector tried finds cones or barriers on WATO
(`docs/perception_2d_measurements.md`), a class never detected is labelled
absent everywhere, and SAM 3's case rested on those classes; perception_2d's
detector-support rules already port its track rules onto SAM2.
