# Pre-downloaded torch wheelhouse

`lidar_preprocessing.Dockerfile` installs torch from this directory via
`uv pip install --no-index --find-links /tmp/wheels torch`.

## Why

Torch + matched NVIDIA CUDA wheels total ~3 GB across ~15 individual wheels
(the torch wheel itself is 700+ MB).  Neither `pip` nor `uv` supports
HTTP range-request resume on partial downloads, so on flaky networks the
build hits connection resets mid-stream and re-fetches the whole wheel each
attempt — often indefinitely.

Running `pip download` on the host is coarse-grained-resumable: completed
wheels persist on disk between attempts.  We trade build-time network access
for one-time host-side fetching that you can retry without losing ground.

## Populating the directory

From the repo root, run:

```bash
./docker/wheels/fetch.sh
```

This loops `pip download` until every wheel torch needs is on disk.  Re-runs
skip already-downloaded files.  Expect ~3 GB of `*.whl` files when done.

## Verifying

```bash
ls docker/wheels/*.whl | wc -l   # expect ~15
du -sh docker/wheels/             # expect ~3 GB
```

## What to do if a torch version bump breaks the build

Delete the existing wheels, update `fetch.sh` to point at the new index/version,
re-run:

```bash
rm docker/wheels/*.whl
./docker/wheels/fetch.sh
```
