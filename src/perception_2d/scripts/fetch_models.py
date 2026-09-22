#!/usr/bin/env python3
"""Fetch ML weights for the perception_2d component.

Run on the host BEFORE launching the container so that ${MODELS_ROOT}
(bind-mounted into the container at /data/models:ro) is populated.

Layout produced:

    ${MODELS_ROOT}/
      sam2.1_hiera_large.pt        # raw SAM2.1 checkpoint (loaded by path)
      hf/                          # HuggingFace cache (HF_HOME)
        hub/models--IDEA-Research--grounding-dino-base/...
        hub/models--depth-anything--Depth-Anything-V2-Large/...
      torch_hub/                   # torch.hub cache (TORCH_HOME)
        hub/checkpoints/dinov2_vitl14_pretrain.pth
        hub/facebookresearch_dinov2_main/...

The perception_2d container expects HF_HOME=/data/models/hf and
TORCH_HOME=/data/models/torch_hub.  Set these in the compose service
environment (or in watod-config.sh) so the runtime loaders find the
pre-downloaded weights.

Usage:
    # Default: write to ./data/models relative to the repo root.
    python3 src/perception_2d/scripts/fetch_models.py

    # Explicit path.
    MODELS_ROOT=/srv/wato_models python3 src/perception_2d/scripts/fetch_models.py

    # Skip a model (e.g. depth if running with depth.enabled: false):
    python3 src/perception_2d/scripts/fetch_models.py --skip depth_anything_v2

Requires (on the host):
    pip install huggingface_hub torch
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry — pinned revisions live in the component package so that this
# host-side fetcher and the in-container runtime loaders read the SAME source.
# Add or re-pin models there, not here.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wato_perception_2d.model_registry import (  # noqa: E402
    ALL_TAGS,
    HF_MODELS,
    RAW_CHECKPOINTS,
    TORCH_HUB_MODELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_models_root(arg: str | None) -> Path:
    if arg:
        root = Path(arg).expanduser().resolve()
    elif "MODELS_ROOT" in os.environ:
        root = Path(os.environ["MODELS_ROOT"]).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parents[3]
        root = (repo_root / "data" / "models").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _du_h(path: Path) -> str:
    """Best-effort human-readable disk usage for a directory."""
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], text=True)
        return out.split()[0]
    except Exception:  # noqa: BLE001
        return "?"


def _fetch_hf(
    repo_id: str, revision: str, hf_home: Path, token: str | None
) -> tuple[bool, str]:
    """Snapshot-download one HuggingFace repo into HF_HOME's hub cache.

    Passes cache_dir explicitly (= HF_HOME/hub) instead of relying on the
    HF_HOME env var: huggingface_hub freezes its cache path at import time, so
    setting os.environ here (after the import) silently falls back to the default
    ~/.cache/huggingface. The container reads HF_HOME=/data/models/hf, so weights
    MUST land in <models_root>/hf/hub to be visible there.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub not installed (pip install huggingface_hub)"

    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=str(hf_home / "hub"),
        )
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _fetch_hf_file(
    repo_id: str, filename: str, revision: str, dest_dir: Path, token: str | None
) -> tuple[bool, str]:
    """Download a single file from an HF repo directly into ``dest_dir``.

    Lands at ``dest_dir/filename`` (a real file, not the HF symlink cache) so a
    runtime loader that takes a plain path finds it. Idempotent: an existing file
    at the destination is left untouched (respects a manual drop).
    """
    dest = dest_dir / filename
    if dest.exists():
        return True, f"already present ({dest})"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False, "huggingface_hub not installed (pip install huggingface_hub)"
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=token,
            local_dir=str(dest_dir),
        )
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _fetch_torch_hub(repo_ref: str, model: str, torch_home: Path) -> tuple[bool, str]:
    """Trigger torch.hub.load to download a model into TORCH_HOME.

    ``repo_ref`` is "owner/repo:<commit>", never a bare branch — the ref also
    names the cache directory (facebookresearch_dinov2_<ref>), so the runtime
    loader must ask for the identical ref to hit this cache instead of the
    network.
    """
    try:
        import torch
    except ImportError:
        return False, "torch not installed (pip install torch)"

    os.environ["TORCH_HOME"] = str(torch_home)
    try:
        torch.hub.load(repo_ref, model, source="github", verbose=False)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download perception_2d model weights to MODELS_ROOT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-root",
        default=None,
        help="Target directory (overrides MODELS_ROOT; default: <repo>/data/models).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="TAG",
        choices=ALL_TAGS,
        help=f"Models to skip.  Choices: {', '.join(ALL_TAGS)}.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (gated repos).  Defaults to $HF_TOKEN.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched and exit.",
    )
    args = parser.parse_args()

    models_root = _resolve_models_root(args.models_root)
    hf_home = models_root / "hf"
    torch_home = models_root / "torch_hub"
    hf_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)

    skip = set(args.skip)

    print(f"MODELS_ROOT = {models_root}")
    print(f"  HF_HOME    = {hf_home}")
    print(f"  TORCH_HOME = {torch_home}")
    print()

    if args.dry_run:
        print("Would fetch:")
        for tag, m in HF_MODELS.items():
            mark = "skip" if tag in skip else "fetch"
            print(f"  [{mark}] {tag:<20} {m.repo_id}@{m.revision[:12]}  → HF cache")
        for tag, c in RAW_CHECKPOINTS.items():
            mark = "skip" if tag in skip else "fetch"
            print(
                f"  [{mark}] {tag:<20} {c.repo_id}@{c.revision[:12]}::{c.filename}"
                f"  → {models_root}"
            )
        for tag, t in TORCH_HUB_MODELS.items():
            mark = "skip" if tag in skip else "fetch"
            print(
                f"  [{mark}] {tag:<20} torch.hub :: {t.repo}@{t.ref[:12]} :: "
                f"{t.entrypoint}"
            )
        return 0

    failures: list[tuple[str, str]] = []

    for tag, m in HF_MODELS.items():
        if tag in skip:
            print(f"⤬ skip   {tag:<20} ({m.repo_id})")
            continue
        print(f"⟶ fetch  {tag:<20} ({m.repo_id}@{m.revision[:12]}) …", flush=True)
        ok, msg = _fetch_hf(m.repo_id, m.revision, hf_home, args.hf_token)
        if ok:
            print("  ✓ ok")
        else:
            print(f"  ✗ {msg}", file=sys.stderr)
            failures.append((tag, msg))

    for tag, c in RAW_CHECKPOINTS.items():
        if tag in skip:
            print(f"⤬ skip   {tag:<20} ({c.repo_id}::{c.filename})")
            continue
        print(
            f"⟶ fetch  {tag:<20} ({c.repo_id}@{c.revision[:12]}::{c.filename}) …",
            flush=True,
        )
        ok, msg = _fetch_hf_file(
            c.repo_id, c.filename, c.revision, models_root, args.hf_token
        )
        if ok:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}", file=sys.stderr)
            failures.append((tag, msg))

    for tag, t in TORCH_HUB_MODELS.items():
        if tag in skip:
            print(f"⤬ skip   {tag:<20} ({t.repo})")
            continue
        repo_ref = f"{t.repo}:{t.ref}"
        print(
            f"⟶ fetch  {tag:<20} (torch.hub :: {repo_ref[:40]}… :: {t.entrypoint}) …",
            flush=True,
        )
        ok, msg = _fetch_torch_hub(repo_ref, t.entrypoint, torch_home)
        if ok:
            print("  ✓ ok")
        else:
            print(f"  ✗ {msg}", file=sys.stderr)
            failures.append((f"{tag}/{t.entrypoint}", msg))

    print()
    print(f"Disk usage  HF_HOME    = {_du_h(hf_home)}")
    print(f"Disk usage  TORCH_HOME = {_du_h(torch_home)}")

    if failures:
        print()
        print(f"⚠ {len(failures)} failure(s):", file=sys.stderr)
        for tag, msg in failures:
            print(f"  - {tag}: {msg}", file=sys.stderr)
        return 1

    print()
    print("All weights fetched.  Set in the container environment:")
    print("  HF_HOME=/data/models/hf")
    print("  TORCH_HOME=/data/models/torch_hub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
