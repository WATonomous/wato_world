#!/usr/bin/env python3
"""Fetch ML weights for the perception_2d component.

Run on the host BEFORE launching the container so that ${MODELS_ROOT}
(bind-mounted into the container at /data/models:ro) is populated.

Layout produced:

    ${MODELS_ROOT}/
      hf/                          # HuggingFace cache (HF_HOME)
        hub/models--facebook--sam3.1/...
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

    # Skip a model (e.g. SAM3 if not yet public on HF Hub):
    python3 src/perception_2d/scripts/fetch_models.py --skip sam3

Requires (on the host):
    pip install huggingface_hub torch
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry.  Edit here when adding / changing the perception_2d stack.
# ---------------------------------------------------------------------------

HF_MODELS: dict[str, str] = {
    # facebook/sam3.1 hosts the multiplex checkpoint (sam3.1_multiplex.pt) +
    # config/tokenizer; loaded via the `sam3` package, not transformers.
    "sam3": "facebook/sam3.1",                          # sam3_concept_tracker.py
    "depth_anything_v2": "depth-anything/Depth-Anything-V2-Large",  # depth.py
}

# DINOv2 weights ship via torch.hub (reid.py).  We pre-populate TORCH_HOME
# by issuing a `torch.hub.load(...)` once.
TORCH_HUB_MODELS: list[tuple[str, str]] = [
    ("facebookresearch/dinov2", "dinov2_vitl14"),
]

ALL_TAGS = list(HF_MODELS.keys()) + ["dinov2"]


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


def _fetch_hf(repo_id: str, hf_home: Path, token: str | None) -> tuple[bool, str]:
    """Snapshot-download one HuggingFace repo into HF_HOME's hub cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, "huggingface_hub not installed (pip install huggingface_hub)"

    os.environ["HF_HOME"] = str(hf_home)
    try:
        snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=None,
            local_dir=None,  # use HF_HOME cache layout
        )
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _fetch_torch_hub(repo: str, model: str, torch_home: Path) -> tuple[bool, str]:
    """Trigger torch.hub.load to download a model into TORCH_HOME."""
    try:
        import torch
    except ImportError:
        return False, "torch not installed (pip install torch)"

    os.environ["TORCH_HOME"] = str(torch_home)
    try:
        torch.hub.load(repo, model, source="github", verbose=False)
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
        help="Target directory (overrides MODELS_ROOT; "
        "default: <repo>/data/models).",
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
        for tag, repo_id in HF_MODELS.items():
            mark = "skip" if tag in skip else "fetch"
            print(f"  [{mark}] {tag:<20} {repo_id}")
        for repo, model in TORCH_HUB_MODELS:
            mark = "skip" if "dinov2" in skip else "fetch"
            print(f"  [{mark}] {'dinov2':<20} torch.hub :: {repo} :: {model}")
        return 0

    failures: list[tuple[str, str]] = []

    for tag, repo_id in HF_MODELS.items():
        if tag in skip:
            print(f"⤬ skip   {tag:<20} ({repo_id})")
            continue
        print(f"⟶ fetch  {tag:<20} ({repo_id}) …", flush=True)
        ok, msg = _fetch_hf(repo_id, hf_home, args.hf_token)
        if ok:
            print(f"  ✓ ok")
        else:
            print(f"  ✗ {msg}", file=sys.stderr)
            failures.append((tag, msg))

    if "dinov2" not in skip:
        for repo, model in TORCH_HUB_MODELS:
            print(f"⟶ fetch  dinov2/{model:<13} (torch.hub :: {repo}) …", flush=True)
            ok, msg = _fetch_torch_hub(repo, model, torch_home)
            if ok:
                print(f"  ✓ ok")
            else:
                print(f"  ✗ {msg}", file=sys.stderr)
                failures.append((f"dinov2/{model}", msg))
    else:
        print(f"⤬ skip   dinov2")

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
    print(f"  HF_HOME=/data/models/hf")
    print(f"  TORCH_HOME=/data/models/torch_hub")
    return 0


if __name__ == "__main__":
    sys.exit(main())
