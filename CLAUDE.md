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
lidar_preprocessing → motion comp, static/dynamic split, ground extraction
perception_2d       → GroundingDINO + SAM2 video tracker + Depth Anything V2 + DINOv2 (optional Florence-2 discovery)
semantic_lifting    → occlusion-aware 2D→3D label lifting (UniLiPs Eq.1)
proposal_generation → LiDAR detector ensemble + Segment-Lift-Fit + fusion
tracking            → 4D tracking with masklet association + DINOv2 ReID
label_refinement    → multimodal LabelFormer (bootstrap → learned)
open_vocab_discovery → rare-class branch
student_training    → BEVFusion / TransFusion student detector
```

`ingest` and `perception_2d` are implemented end-to-end. `semantic_lifting` core
algorithm is implemented (Parts 1–7). Everything else is a stub that raises
NotImplementedError.

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

Single command: `./watod run ingest <bag>` invokes
`python -m wato_ingest run --bag <bag>` which calls
`wato_ingest.pipeline.run_bag()`. That orchestrates:

```
inputs/        decoders/         artifacts/
bags.py        cameras.py        frame_index.py
calibration.py lidar.py          quality.py
chunks.py      poses.py          manifest.py
topics.py      pose_interpolation.py
```

Outputs land at `data/artifacts/raw/<bag_id>/` per the schema in
`config/postgres/init.sql` (legacy filename — Postgres is gone, file just
documents the table shapes the artifacts use). `frame_index.parquet` is the
contract every downstream component reads.

**Pose source**: configurable via `topics.pose` in
`src/ingest/config/ingest.yaml`. Canonical source is eidos's `slam/odometry`
(stamped at LiDAR keyframe sensor time, map frame, child=base_footprint).
Do NOT consume `/tf` directly — eidos doesn't publish to it; eidos_transform
does, and that stream is wall-clock-stamped which desyncs from LiDAR.

**Calibration**: auto-extracted from the bag itself. `freeze_from_bag()` reads
`/<cam>/camera_info` (intrinsics, distortion, frame_id), first PointCloud2
per LiDAR (frame_id), and the configured TF topic (extrinsics via BFS chain).
`--calibration <file>` overrides with a hand-authored JSON.

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
PYTHONPATH=src/common/src:src/ingest/src python3 -m pytest src/ingest/tests
# 31 passing tests, all without ROS installed (lazy ROS imports in rosbag_reader).
```
