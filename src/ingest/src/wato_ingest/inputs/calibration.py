"""Freeze a per-bag calibration JSON.

Two ingest modes:
1. `from_file` — copy an existing calibration JSON authored elsewhere into the
   bag's artifact root (the common case until calibration tooling lives here).
2. `from_camera_info` — derive intrinsics from /<cam>/camera_info messages in
   the bag and merge with extrinsics from /tf_static.

Either way, the output is `raw/<bag_id>/calibration.json` with the schema:
    {
      "calibration_version": str,
      "cameras": {
        "<cam_id>": {"K": 3x3, "distortion": [...], "width": int, "height": int,
                     "ego_T_cam": 4x4}
      },
      "lidars": {"<lidar_id>": {"ego_T_lidar": 4x4}},
      "checks": {"sanity": "ok" | "skipped" | "failed", "notes": str}
    }
"""

from __future__ import annotations

import json
import os
import shutil

from wato_common.artifact_store import (
    bag_root,
    calibration_path,
    ensure_local_dir,
    local_path,
)


def freeze_from_file(bag_id: str, source_path: str, *, version: str | None = None) -> str:
    """Copy an authored calibration JSON to the bag's artifact root.

    The source is validated (must contain `cameras` and `lidars` keys) and
    annotated with `calibration_version` if provided.  Returns the URI written.
    """
    with open(source_path, "r", encoding="utf-8") as fh:
        calib = json.load(fh)

    if "cameras" not in calib or "lidars" not in calib:
        raise ValueError(
            f"calibration {source_path} missing required keys 'cameras'/'lidars'"
        )
    if version:
        calib["calibration_version"] = version
    calib.setdefault("checks", {"sanity": "skipped", "notes": "imported from file"})

    out_uri = calibration_path(bag_id)
    ensure_local_dir(os.path.dirname(local_path(out_uri)))
    with open(local_path(out_uri), "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2)
    return out_uri


def load(bag_id: str) -> dict:
    with open(local_path(calibration_path(bag_id)), "r", encoding="utf-8") as fh:
        return json.load(fh)


def sanity_check(bag_id: str) -> tuple[bool, str]:
    """Stub.  Returns (passed, notes).

    A full check projects LiDAR points into each camera and verifies the
    distribution lands inside the image with reasonable density.  Implement
    once ingest produces a sample sweep + image pair.
    """
    try:
        calib = load(bag_id)
    except FileNotFoundError:
        return False, "calibration.json missing"
    if "cameras" not in calib:
        return False, "no cameras in calibration"
    return True, "structural check only; numeric sanity not yet implemented"
