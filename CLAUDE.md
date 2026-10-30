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

Offline batch 3D auto-labeling for WATonomous: nine components, one Docker
image each, communicating only through files under `data/artifacts/`
(Parquet / JSON / PNG / NPZ). No runtime messaging, no database.

```
ingest → lidar_preprocessing → perception_2d → semantic_lifting →
proposal_generation → tracking → label_refinement / open_vocab_discovery → student_training
```

ingest, lidar_preprocessing, perception_2d and semantic_lifting are implemented
end to end; the rest are stubs raising NotImplementedError. Each implemented
component's README is its reference.
Measurements behind defaults are in `docs/*_measurements.md`; which recordings
are usable is `docs/drive_bag_survey.md`.

## Conventions

- Layout: `src/<component>/src/wato_<component>/`, tests in
  `src/<component>/tests/`, config in `src/<component>/config/`; shared code in
  `src/common/src/wato_common/`.
- **One module per cross-component artifact.** Anything one component writes
  and another reads is encoded, written and read by its module in
  `wato_common/artifacts/` (`calibration`, `lidar_sweeps`, `motion_proposals`,
  `masklets`, `depth`, `lifted_labels`), round-trip tested in
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
./watod run <component> --bag <bag> [flags]   # also: validate-bag, trim-bag, reduce, viz
./watod test <component>      # src/common/tests, then the component's, in its dev image
./watod down all
./watod build-base            # only when ghcr lacks the bases (first-time setup)
```

`ACTIVE_MODULES` matches the monorepo: `"ingest:dev"`, `"all"`, `"all:dev"`, or
a list; `:dev` = develop target with source bind-mounts. Run tests with
`./watod test`: the images have pyarrow, ROS, numba and pypatchworkpp. On a host
with the Python deps,
`PYTHONPATH=src/common/src:src/ingest/src python3 -m pytest -p no:anyio src/ingest/tests`
runs ingest's suite without ROS (`-p no:anyio`: the host's anyio plugin
doesn't load; the trim-bag end-to-end test skips).

## Cross-component contracts

- **`frame_index` has one row per (LiDAR sweep, camera)**, so an image repeats
  once per sweep it is nearest to (~4.9× on the 3-LiDAR WATO rig). Per-image
  work collapses rows by `(cam_id, camera_seq)` first: perception_2d
  `_one_frame_per_image` (keeping the matched `sweep_ids` for depth anchors),
  semantic_lifting `load_sweep_frames` (one shared frame per image, its depth
  and mask raster cached across the sweeps it is paired with). Feeding rows straight to SAM2 got the
  process SIGKILLed on a 30 s WATO chunk.
- **Every pose is looked up at its datum's own time**: a sweep's, a deskewed
  point's, a camera frame's, all interpolated from `poses.parquet` through
  `wato_common.pose_lookup.PoseLookup`, which honours ingest's per-stretch
  verdicts. `frame_index.world_T_ego` is the SWEEP's pose; projecting into an
  image uses `camera_timestamp_ns` (up to 44 ms apart on WATO, 120 ms on
  nuScenes).
- **`sweep_id` is one counter per chunk across all LiDARs**, so per-sweep files
  and joins key on `sweep_id` alone. Chunks ingested before that restart it per
  LiDAR (the three WATO scanners overwrite each other's world files) and must be
  re-ingested.
- **lidar_preprocessing has two dynamic artifacts.** `*_dynamic_mask.npy` /
  `dynamic_map.npz` is the seg method's *precision* verdict (`~mask` is trusted
  static: perception_2d's depth anchors), read through
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
  of `global_object_id`) runs after semantic_lifting, so lifting merges one
  object's per-camera masklets itself from the LiDAR points they both claim;
  `lifted_labels` name the instance by its smallest masklet_id and
  `lifted_instances.parquet` lists its masklets and cameras.
- **Sensor physics lives in `sensor_model.py`** (profiles `vlp32c`, `vlp16`,
  `hdl32e`), never in YAML.

## Ingest

Details: `src/ingest/README.md`; measurements: `docs/ingest_measurements.md`.

- **Dataset-agnostic: a new recording needs only a new YAML.** Nothing in
  `wato_ingest` names a dataset, vehicle or topic (the `wato_` prefix is the
  repo namespace). A recording's quirk goes in its profile YAML or is rejected
  with a named error. Ingest validates topic types up front, detects the
  rosbag2 storage plugin, and stops rather than write artifacts that look
  complete but aren't.
- **Profiles.** `ingest.yaml` (nuScenes); `ingest.wato.yaml` (8 rectified pano
  cameras, per-corner LiDARs, eidos `liso/odometry`, `ego_frame:
  base_footprint`); `ingest.wato_6cam.yaml` = `ingest.wato.yaml` without
  `PANO_EE` / `PANO_WW` (`ring_road_corrected`, May 30 tests 2–3); a test fails
  if any other value drifts. lidar_preprocessing's `.wato.yaml` pairs with them
  (`per_lidar`: `lidar_cc` vlp32c, `lidar_ne`/`lidar_nw` vlp16;
  `frame_sync.canonical_lidar: lidar_cc`; `header_stamp_at: sweep_end`, from the
  car's `timestamp_first_packet: false`, since WATO Velodyne `time` fields are
  all zero and deskew synthesizes times from azimuth).
- **WATO LiDAR: ingest the per-corner topics**, not `/lidar/all/points_merged`:
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
  `distortion = []` (raw CameraInfo kept). Consumers just project with `K`
  through `wato_common.artifacts.calibration`, whose accessors raise on a
  missing extrinsic.
- **Before ingest, `validate-bag` / `trim-bag`.** Judged on log time. trim-bag
  uses the pinned `mcap` CLI (rosbag2's writer would empty custom-type schemas)
  and writes to `data/bags/trimmed/`, the only read-write mount under
  `/data/bags`. One-shot eidos topics must be passed as `--static-topic`, or
  there is no common window.

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
| Python deps | `docker/requirements/{ingest,lidar_preprocessing,perception_2d,semantic_lifting}.txt` | exact `==` for the full transitive closure, installed `--no-deps` |
| `sam2` | `docker/perception_2d.Dockerfile` | `ARG SAM2_COMMIT` (no PyPI release) |
| MF-MOS | `docker/lidar_preprocessing.Dockerfile` | `ARG MF_MOS_COMMIT` |
| Model weights | `src/perception_2d/src/wato_perception_2d/model_registry.py` | HF `revision=`, torch.hub `repo:<sha>`, raw `.pt` from an HF commit loaded by path |
| Component bases | `docker/{ingest,lidar_preprocessing,perception_2d,semantic_lifting}.Dockerfile` | `base:<tag>@sha256:...` |
| Upstream bases | `docker/base/inject_*.Dockerfile` | `ubuntu` / `nvcr.io` by digest, `ARG UV_VERSION` |
| `mcap` CLI | `docker/ingest.Dockerfile` | `ARG MCAP_CLI_VERSION` + `MCAP_CLI_SHA256` |

Unpinned: the other five Dockerfiles, all stubs (mutable `base:<tag>`, no
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
`sort docker/requirements/<c>.txt | grep -v '^#'`. The two GPU components need
`--index-strategy unsafe-first-match`: `download.pytorch.org/whl/cu128` mirrors
common packages at its own versions, and uv's default then refuses PyPI.

**Provenance** (`wato_common/`): `provenance.py` reads `WATO_GIT_COMMIT`,
`WATO_GIT_DIRTY`, `WATO_BUILD_TIME`, `WATO_BASE_IMAGE` and the hash of
`/opt/watonomous/requirements.lock.txt`; `manifest.py` writes one manifest per
(bag, chunk) per component with content-hashed inputs (> 64 MB: size + mtime),
so a changed input marks the outputs stale. ingest writes `manifest.json`;
lidar_preprocessing, perception_2d and semantic_lifting write
`manifest_<component>.json` (semantic_lifting hashes its upstreams' manifests
in place of their per-sweep files).

- `WATO_GIT_COMMIT` is baked in at build time (`watod-setup-env.sh` →
  `modules/.env` → build arg): deploy images have no `.git`.
- `ARG BASE_IMAGE` is global; re-declare it inside the `dependencies` stage or
  `ENV WATO_BASE_IMAGE=${BASE_IMAGE}` expands to empty.
- Build `_source` profiles before the runtime profile: one `docker compose
  build` runs them in parallel and ships stale code. `watod build` orders them.

Known drift frozen into the locks, to fix as separate testable changes:
lidar_preprocessing floated to `torch==2.11.0` + the multi-GB `cuda-toolkit`
wheel (perception_2d pins 2.7.1 to avoid it); perception_2d declares `numpy<2`
but ships 2.5.3; perception_2d has both `opencv-python` and
`opencv-python-headless`; `depth-anything-v2` pulls in gradio, fastapi, uvicorn,
starlette.

## Pre-commit gotchas

- `PermissionError: '.codex'`: a deleted file still in the index — `git rm .codex`.
- F821 `Undefined name 'reader'` in `rosbag_reader.py` comes from a `del reader`
  in the contextmanager's `finally`; leave the cleanup to refcounting.

## Removed on purpose

Postgres / any database (`psycopg`, `sqlalchemy`, `PG_DSN`, `PG_PORT`,
`docker-compose.infra.yaml` references are stale; the table shapes are
`wato_common.schemas`), `watod bag` / `watod-bag.sh`,
`watod_completion.bash` (port it from the monorepo if wanted),
`watod-config.local.sh.example`.
