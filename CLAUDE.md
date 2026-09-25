# wato_world — Claude project guide

## Prime directive: mirror `wato_monorepo`

**Everything in this repo — architecture, infrastructure, code style, naming,
shell scripts, Dockerfiles, CI workflows, directory layout — should be derived
from or copied from `wato_monorepo`** (sibling clone at
`/home/brianzheng/wato_world/wato_monorepo`).

When in doubt:
1. Look at how the monorepo solves the equivalent problem.
2. Copy that pattern, adapting only where wato_world's offline-batch
   nature genuinely differs from the monorepo's runtime ROS pipeline.
3. Don't invent new patterns. Invented infrastructure has a track record of
   needing to be ripped out and re-aligned with the monorepo later.

Concrete reference points:
- `wato_monorepo/watod` — entrypoint shape, flag parsing, `ACTIVE_MODULES`
  with `:dev` suffix.
- `wato_monorepo/watod_scripts/watod-compose.sh` — pull-bases-then-build
  flow on `build`.
- `wato_monorepo/docker/base/inject_*.Dockerfile` — `ARG GENERIC_IMAGE`
  injection layer pattern.
- `wato_monorepo/docker/template.Dockerfile` — multi-stage `source` /
  `dependencies` / `deploy` / `develop` shape.
- `wato_monorepo/.github/workflows/build_base_images.yml` — CI publish
  pattern for base images.
- `wato_monorepo/.pre-commit-config.yaml` — lint/format hooks.

Where wato_world legitimately diverges:
- No runtime ROS messaging → no Zenoh, no `network_mode: host`, no
  `package.xml`/`CMakeLists.txt`, no colcon. Components communicate via
  artifacts on disk.
- No `infrastructure` profile (no shared always-on services in compose).
- No carla/simulation profiles.
- Per-component config lives at `src/<component>/config/<component>.yaml`
  (component-owned) rather than a single root `config/pipeline.yaml`.

If you're about to write a shell script, Dockerfile, workflow YAML, or
helper, **first grep the monorepo for the equivalent and copy its shape.**

## What this repo is

Offline batch 3D auto-labeling pipeline for WATonomous. Nine components,
each in its own Docker image, communicating only through artifacts on disk
(Parquet/JSON/PNG/NPZ). No runtime ROS messaging, no database.

```
ingest              → frames + lidar sweeps + poses + frame_index
lidar_preprocessing → motion comp, static/dynamic split (--seg aw|mos|union), ground, bag IWU, motion proposals
perception_2d       → GroundingDINO + SAM2 video tracker + Depth Anything V2 + DINOv2 (optional Florence-2 discovery)
semantic_lifting    → occlusion-aware 2D→3D label lifting (UniLiPs Eq.1)
proposal_generation → LiDAR detector ensemble + Segment-Lift-Fit + fusion
tracking            → 4D tracking with masklet association + DINOv2 ReID
label_refinement    → multimodal LabelFormer (bootstrap → learned)
open_vocab_discovery → rare-class branch
student_training    → BEVFusion / TransFusion student detector
```

`ingest`, `lidar_preprocessing`, and `perception_2d` are implemented
end-to-end. `semantic_lifting` core algorithm is implemented (Parts 1–7).
Everything else is a stub that raises NotImplementedError.

**Dataset profiles**: `ingest` and `lidar_preprocessing` each ship two config
profiles — the default (nuScenes) and a `.wato.yaml` variant for the
3-Velodyne rig (per-corner lidar topics, per-lidar sensor profiles —
`vlp32c` for `lidar_cc`, `vlp16` for `lidar_ne`/`lidar_nw` — and
`frame_sync.canonical_lidar: lidar_cc`). Valid profiles are `vlp32c`, `vlp16`,
`hdl32e`; physics lives in `sensor_model.py`, not YAML. Driver properties live
in the profile YAML instead: WATO Velodyne per-point `time` fields are all
zero, so deskew treats them as missing and synthesizes times from azimuth,
counting back from the stamp (`header_stamp_at: sweep_end`, from the car's
`timestamp_first_packet: false`). Ingest's `sweep_id` is one counter per chunk
shared by all LiDARs, which is what lets lidar_preprocessing, perception_2d and
semantic_lifting key per-sweep files and joins on `sweep_id` alone; chunks
ingested before that restart it per LiDAR and must be re-ingested (their three
WATO scanners overwrite each other's world files). WATO bags must be ingested with the
per-corner topics, NOT `/lidar/all/points_merged`: classify's ray traversal
assumes one sensor origin per sweep, and merged clouds leak static structure
into `dynamic_map.npz`.

**lidar_preprocessing has two dynamic artifacts — don't conflate them.**
`*_dynamic_mask.npy` / `dynamic_map.npz` is the chosen seg method's
*precision* verdict (perception_2d's depth anchors read `~dynamic_mask` as
trusted static; semantic_lifting carries the path for its planned
dynamic-point handling). `*_motion_proposals.npz` /
`motion_clusters.parquet` (Step F) is the *recall* artifact: every heuristic's
per-point verdict (`source_bits`: AW dynamic/ambiguous, IWU-evicted, MF-MOS,
seg dynamic, box fill, unmapped) plus HDBSCAN clusters with soft motion
features (Chen et al.'s motion_score). False positives there are by design —
downstream association filters them. Step E (`iwu/`, bag-level UniLiPs IWU →
`global_iwu.npz`) evicts floaters from the bag static map and feeds Step F.
The box / Kalman / association code Step F tracks with lives in
`wato_common/tracking/` so the `tracking` component reuses it rather than
growing a second motion model. `--two-pass` (a one-time global-map log-odds
prior) is off by default and is NOT IWU.

## Repository conventions

- **Mirrors `wato_monorepo` patterns.** When in doubt, look at how the monorepo
  does it. Don't invent.
- Top-level `watod` script + `watod_scripts/` helpers + `watod-config.sh`
  user config. Same shape as the monorepo's.
- Components live at `src/<component>/src/wato_<component>/`. Tests at
  `src/<component>/tests/`. Component-owned config at `src/<component>/config/`.
- Shared Python lib at `src/common/src/wato_common/` (geometry, schemas,
  artifact_store, io/{parquet,pointcloud2,rosbag_reader}).
- Docker: per-component Dockerfile at `docker/<component>.Dockerfile`,
  base injection layers at `docker/base/inject_{cpu,cuda}_base.Dockerfile`.
- Compose stack: `modules/docker-compose.yaml` (deploy) + `.dev.yaml` (dev) +
  `.gpu.yaml` (gpu host overrides).
- **Keep docs in lockstep with code.** Every time you add or change a feature,
  update the relevant documentation in the same change — the component
  `README.md`, this `CLAUDE.md`, config comments, and any docstrings describing
  the affected behavior. A change that alters observable behavior, a contract
  (artifact schema, `frame_index` fields, flags), or a cross-component
  assumption is not complete until the corresponding markdown reflects it.
  When you touch a component, verify its `README.md` still matches reality and
  fix any drift you find.

## watod CLI

```bash
# Edit ACTIVE_MODULES in watod-config.sh, e.g. "ingest:dev"
./watod build         # pulls bases from ghcr, builds component images
./watod up            # docker compose up -d
./watod -t ingest_dev # bash shell into the running dev container
./watod run ingest /data/bags/<bag>.mcap
./watod test ingest
./watod down all
./watod build-base    # ONLY needed if ghcr lacks the bases (first-time setup)
```

`ACTIVE_MODULES` syntax matches monorepo: `"ingest:dev"`, `"all"`, `"all:dev"`,
or a list. `:dev` = develop target with source bind-mounts.

## Ingest pipeline (the only complete component)

**Dataset-agnostic: a new recording needs only a new YAML, never Python.**
Nothing in `wato_ingest` names a dataset, vehicle or topic (the `wato_` prefix
is the repo namespace every package carries, like `wato_common`). Ingest
validates the configured topics' message types up front, detects the rosbag2
storage plugin itself, and stops with a named error rather than writing
artifacts that look complete but aren't (ingest README "Ingesting a new bag").
Keep it that way: a quirk of one recording belongs in its profile YAML or is
rejected with a clear error, not special-cased in code.

Single command: `./watod run ingest <bag>` invokes
`python -m wato_ingest run --bag <bag>` which calls
`wato_ingest.pipeline.run_bag()`. That orchestrates:

```
inputs/        decoders/         artifacts/
bags.py        cameras.py        frame_index.py
calibration.py lidar.py          quality.py
chunks.py      poses.py          manifest.py
topics.py
```

Outputs land at `data/artifacts/raw/<bag_id>/` per the schema in
`config/postgres/init.sql` (legacy filename — Postgres is gone, file just
documents the table shapes the artifacts use). `frame_index.parquet` is the
contract every downstream component reads.

**Pose lookup — one place, at each datum's own time**: every pose (a sweep's,
a deskewed point's, a camera frame's) is interpolated from `poses.parquet`
through `wato_common.pose_lookup.PoseLookup`, called at the timestamp of the
data being placed. `frame_index.world_T_ego` is the SWEEP's pose; anything
projecting into an image looks the pose up at `camera_timestamp_ns` instead
(perception_2d depth anchors, semantic_lifting) — they're up to 44 ms apart on
WATO, 120 ms on nuScenes.

**Pose source — must be dense and smooth**: `topics.pose` (one
`nav_msgs/Odometry` topic whose `child_frame_id` equals `ego_frame`, or a
`PoseStamped` / `PoseWithCovarianceStamped`, whose body frame can't be checked). Every
pose is linearly interpolated between two samples, so ingest enforces
`pose_requirements` (see ingest README "Pose requirements"): a chunk whose
pose span is < `min_dense_fraction` covered by samples ≤ `max_bracket_ms`
apart ABORTS ingest; stretches between distant samples or across a jump are
marked in `poses.parquet` (`interval_drop_reason`), and nothing inside them
gets a valid pose (sweeps: `valid_pose=False` + `pose_drop_reason`). Passing the
checks is necessary, not sufficient — they can't see a mis-stamped or
attitude-biased stream (ingest README "How good are the poses"). Eidos topics:
`liso/odometry` = per-scan front-end pose stamped at scan time (agrees with the
INS to ~5 cm/s; no ground truth exists); `slam/odometry` = back-end
newest-KEYFRAME pose — one per 5 m on eidos main (aborts); in
ring_road_corrected it is bit-identical LISO poses stamped 244–450 ms late (median 310 ms; LISO processing latency + wait for the next SLAM tick, so no constant shift corrects it).
Eidos main's GPS/loop-closure-corrected keyframe poses were never compared to
LISO (no recording has both). WATO profile: `ingest.wato.yaml` (eidos
`slam/odometry`, child `base_footprint`). No NovAtel profile ships:
`/novatel/oem7/odom` (UTM, child `base_link`, 50 Hz position) is NOT validated
for labeling — ~2° attitude bias vs the LiDAR frame, mount vs INS error
undetermined. Do NOT consume `/tf` directly — eidos
doesn't publish to it; eidos_transform does, and that stream is
wall-clock-stamped which desyncs from LiDAR.

**Calibration**: auto-extracted from the bag itself. `freeze_from_bag()` reads
each camera's `info` topic (intrinsics, distortion, frame_id), first PointCloud2
per LiDAR (frame_id), and the configured TF topic (extrinsics via BFS chain).
`--calibration <file>` overrides with a hand-authored JSON (the bag's `info` /
`tf_static` topics are then not required). Either way `require_complete()`
aborts ingest if a configured sensor has no intrinsics or extrinsics.

## CI / ghcr — the lessons that took several iterations to learn

### The bootstrap pattern matches the monorepo

CI publishes base images to ghcr; locals pull. There's no
`watod-build-base.sh` in the monorepo because their CI has been working for
years. wato_world has the same shape but on a fresh repo, so first-time
bootstrap requires the workflow to actually run successfully once.

### Three workflows

- `.github/workflows/build_base_images.yml` — builds + pushes the two base
  images (`base:cpu-ubuntu24.04`, `base:cuda12.8.1-cudnn-runtime-ubuntu24.04`)
  to ghcr. Triggers on:
  push to main with `docker/base/**` changes, OR `workflow_dispatch`.
- `.github/workflows/build_and_test.yml` — builds each component matrix
  entry against the published bases and runs pytest. Triggers on PRs and
  pushes to main. **Will fail with `not found`/`403` until base images
  exist on ghcr.**
- `.github/workflows/pre-commit.yml` — lint/format gate.

### Required setup (do these once)

1. **Repo Settings → Actions → General → Workflow permissions = "Read and
   write permissions"**. Without this, the auto-`GITHUB_TOKEN` lacks
   `packages: write` even when the workflow declares it.
2. **Push the workflow files to `main`.** GH only discovers workflows from
   the default branch.
3. **Trigger `build_base_images` manually** the first time:
   GH UI → Actions → "build-base-images" → Run workflow → branch `main`.
4. **Wait for both matrix jobs (cpu + cuda) to finish green.** Verify a log
   line like `pushing manifest for ghcr.io/watonomous/wato_world/base:...`.
5. **Package → repo association is now automatic** via the
   `org.opencontainers.image.source` label set in `build_base_images.yml`.
   ghcr reads that label on push and links the package to the repo, so
   downstream workflows' `GITHUB_TOKEN` (with `packages: read`) can pull
   without further action. If the label was missing on the first push and
   you see 403s, either republish the bases (any push to `docker/base/**`)
   OR fix manually: package Settings → Change visibility → Public, OR
   "Manage Actions access" → Add Repository → `wato_world` → Read.
6. From this point on, every push to `docker/base/**` auto-republishes;
   every PR's `build_and_test` pulls the published bases.

### Gotchas that already burned us once

- **`${{ github.repository }}` preserves case** (`WATonomous/wato_world`) but
  ghcr tags must be lowercase. The workflow has a `repo` step that runs
  `tr '[:upper:]' '[:lower:]'`. Don't simplify away.
- **`ARG GENERIC_IMAGE` mirrors the monorepo's base injection pattern.**
  Build with `docker build --build-arg GENERIC_IMAGE=ubuntu:24.04` (or the
  CUDA variant). The local `watod-build-base.sh` does this.
- **403 vs `not found`**: ghcr returns 403 for non-existent packages when
  queried anonymously, and `not found` when authenticated. If you see 403
  in CI logs, you're missing `docker/login-action` (or the package doesn't
  exist). If you see `not found` after auth is set up, the package wasn't
  published.
- **Ubuntu 24.04 ships Python 3.12, not 3.11.** Both base Dockerfiles use
  `python3` (no version pin). Don't pin to 3.11 — apt will fail.
- **`build_and_test.yml` MUST have `docker/login-action`** before the
  build-push step, even if packages are public, because buildx tries
  anonymous-token first and ghcr 403's it for org-owned packages.
- **Don't auto-build bases on every `watod build`.** The monorepo doesn't.
  We pull from ghcr; `watod-build-base` is only a manual fallback for
  first-time setup or offline iteration on the base Dockerfiles.

### Quick diagnostic commands

```bash
# Did the bases publish?
docker pull ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04   # should succeed
# If "not found": build_base_images hasn't published yet.
# If "denied"/"403": package is private + you're unauthenticated.

# Force a clean local pull (ignores cache):
docker rmi ghcr.io/watonomous/wato_world/base:cpu-ubuntu24.04
./watod build  # should print "Pulling..." with no "Skipped" line

# Authenticate locally if packages are private:
echo $GH_PAT | docker login ghcr.io -u <username> --password-stdin
# PAT needs at least: read:packages
```

## Reproducibility — pinning and provenance

**The pipeline's output is training data.** A rebuild that silently resolves a
different dependency set, model revision, or base image is a *different
labeler*, and its labels are not comparable to the previous run's. Everything
below exists to make that impossible to do by accident.

### What is pinned

| Surface | Where | Mechanism |
|---|---|---|
| Python deps | `docker/requirements/{ingest,lidar_preprocessing,perception_2d}.txt` | exact `==` for the full transitive closure, installed `--no-deps` |
| `sam2` | `docker/perception_2d.Dockerfile` | `ARG SAM2_COMMIT` — upstream has no PyPI release, `main` moves |
| MF-MOS | `docker/lidar_preprocessing.Dockerfile` | `ARG MF_MOS_COMMIT` — decides the static/dynamic split |
| Model weights | `src/perception_2d/src/wato_perception_2d/model_registry.py` | HF `revision=` / torch.hub `repo:<sha>` |
| Component bases | `docker/{ingest,lidar_preprocessing,perception_2d}.Dockerfile` | `base:<tag>@sha256:...` |
| Upstream bases | `docker/base/inject_*.Dockerfile` | `ubuntu`/`nvcr.io` by digest, plus `ARG UV_VERSION` |

**Only the three implemented components are pinned.** The other six
Dockerfiles — including `semantic_lifting`, whose core algorithm is
implemented — still use a mutable `base:<tag>`, have no lockfile, set no
`WATO_BASE_IMAGE`, and write no manifest. A CI republish of the base silently
changes them. Pin each one (digest + lockfile + manifest) when it stops being
a stub.

Also not pinned: the ROS 2 Jazzy **apt** packages in `ingest`. apt has no
lockfile. Known gap, documented at the bottom of `docker/requirements/ingest.txt`.

### Regenerating a lockfile

Lockfiles are **generated, not hand-edited**. Change the intent comment in the
component Dockerfile, build the deps stage, then re-freeze:

```bash
docker run --rm --entrypoint uv \
    ghcr.io/watonomous/wato_world/<component>:deps_<tag> pip freeze --system \
    | grep -v '^Using Python'
```

Commit the Dockerfile change and the lockfile together. Bumping a pin by hand
without rebuilding is wrong — the transitive set moves with it.

**`--index-strategy unsafe-first-match`** is required for the two GPU
components. uv's default considers only the first index carrying a package
name, and `download.pytorch.org/whl/cu128` mirrors common packages (certifi,
etc.) at its own versions, so the default refuses to fall back to PyPI and the
resolve fails. Rationale is recorded in the lockfile headers — don't remove it.

### Provenance — what actually produced an artifact

Pinning only makes a rebuild deterministic; it can't tell you *which* pinned
configuration produced the labels in front of you. That's `wato_common/`:

- `provenance.py` — reads build-time facts baked into the image as env vars
  (`WATO_GIT_COMMIT`, `WATO_GIT_DIRTY`, `WATO_BUILD_TIME`, `WATO_BASE_IMAGE`)
  plus a hash of the lockfile kept at `/opt/watonomous/requirements.lock.txt`.
- `manifest.py` — one manifest per (bag, chunk) per component, recording
  provenance, **content-hashed inputs**, and outputs.

Input hashes are what make staleness detectable: if perception_2d's manifest
records the `frame_index.parquet` it consumed and that hash no longer matches,
its outputs are stale. Files over 64 MB record size+mtime instead — hashing
every lidar sweep would cost minutes per chunk.

Manifest filenames: ingest writes `manifest.json` (historical);
lidar_preprocessing and perception_2d write `manifest_<component>.json` into
the same chunk directory. semantic_lifting and the stubs write none yet.

Gotchas that already cost time here:

- **`WATO_GIT_COMMIT` must be baked in at build time**, not discovered at
  runtime. Deploy images have no `.git`, so `git rev-parse` in a container
  silently returns `""`. `watod_scripts/watod-setup-env.sh` writes it to
  `modules/.env`; compose passes it as a build arg.
- **`ARG BASE_IMAGE` must be re-declared inside the `dependencies` stage.**
  It's a global ARG (declared before the first `FROM`), and global args are
  not in scope inside a stage until re-declared — without the bare
  `ARG BASE_IMAGE`, `ENV WATO_BASE_IMAGE=${BASE_IMAGE}` expands to empty.
- **Build `_pre` profiles before the runtime profile.** Building
  `<comp>_source` and `<comp>` in one `docker compose build` runs them in
  parallel, so the template stage resolves `FROM ${MODULE_SOURCE}` against the
  *previous* source image and silently ships stale code. `watod build` does
  the two phases in order; ad-hoc compose invocations must too.

### Verifying a lock still reproduces

```bash
docker run --rm --entrypoint uv <image>:deps_<tag> pip freeze --system \
    | grep -v '^Using Python' | sort | diff - <(sort docker/requirements/<c>.txt | grep -v '^#')
```
All three implemented components were verified byte-identical after locking.

### Known drift frozen into the current locks

Locking captured a set that *works*, which also froze some accidents. These are
documented in each lockfile header and are worth cleaning up as a separate,
testable change — not by hand-editing a lock:

- `lidar_preprocessing` floated to `torch==2.11.0` and pulled the multi-GB
  `cuda-toolkit` meta-wheel that `perception_2d` pins `torch==2.7.1`
  specifically to avoid. The two GPU components are on different
  torch/cuDNN/NCCL builds.
- `perception_2d` declares `numpy<2` but ships `numpy==2.5.3` — a later layer
  upgraded it and the stated constraint (for the since-removed `sam3`) no
  longer holds.
- `perception_2d` has both `opencv-python` and `opencv-python-headless`.
- `depth-anything-v2` drags in `gradio`, `fastapi`, `uvicorn`, `starlette` —
  a web-app stack inside a batch labeler.

## Pre-commit gotchas

- **`PermissionError: '.codex'`**: phantom git entry — file deleted from
  disk but still in index. Fix: `git rm .codex`.
- **F821 `Undefined name 'reader'` in rosbag_reader.py**: caused by
  `del reader` in the contextmanager `finally` block, which ruff flags
  even though Python's closure resolution makes it work. Fix: just remove
  the `del`; CPython refcounting handles the cleanup when the closure dies.

## What's NOT in this repo (deliberately)

- **Postgres / database.** Removed — artifact tree IS the metadata index.
  Anything still referencing `psycopg`, `sqlalchemy`, `PG_DSN`, `PG_PORT`,
  or `docker-compose.infra.yaml` is stale.
- **`watod-bag.sh` / `watod bag` subcommand.** Not needed for offline batch.
- **`watod_completion.bash`.** Removed; port from monorepo if anyone wants
  tab completion.
- **`watod-config.local.sh.example`.** Removed; the user config file
  itself documents overrides.

## Build / test smoke check

```bash
PYTHONPATH=src/common/src:src/ingest/src python3 -m pytest -p no:anyio src/ingest/tests
# 79 passing tests, all without ROS installed (lazy ROS imports in rosbag_reader).
# `-p no:anyio`: the host's anyio pytest plugin doesn't load under its pytest.
```
