"""Per-camera mask diagnostics, written when ``segmentation.mask_quality`` is on.

perception_2d-local, not a cross-component artifact: no other component reads
it (scripts/diagnose_masks.py does). One JSON per camera,
``raw/<bag>/chunks/<chunk>/mask_quality/<cam_id>.json``, written once the
camera's tracking has succeeded; ``--force`` clears the directory.

``events`` are sam2_tracker.track_camera's ``diagnostics``: a "mask" event per
(frame, object) with raw, postprocessed and saved pixel areas, threshold
stability, ended / tentative flags and, under DAM4SAM, its memory frames and
SAM2's own confidence (``predicted_iou`` of the chosen mask,
``object_score_logit``); a "birth" event per box the birth gate tested, with
its untracked fraction or mask IoU and whether it started a track; a
"backfill" event per back-filled track, with the masks added, each one's
predicted IoU and why it stopped; a "prompt" event per refinement prompt, with the image predictor's
scores and whether it was applied. Stability is area(logit > +offset) /
area(logit > −offset): threshold sensitivity, not accuracy or temporal
consistency; predicted IoU and object score are model confidence too.
"""

from __future__ import annotations

import json
import os

from wato_common.artifact_store import chunk_root, local_path

SCHEMA_VERSION = 2


def directory_uri(bag_id: str, chunk_id: str) -> str:
    return f"{chunk_root(bag_id, chunk_id)}/mask_quality"


def directory(bag_id: str, chunk_id: str) -> str:
    return local_path(directory_uri(bag_id, chunk_id))


def path(bag_id: str, chunk_id: str, cam_id: str) -> str:
    return os.path.join(directory(bag_id, chunk_id), f"{cam_id}.json")


def write(
    bag_id: str,
    chunk_id: str,
    cam_id: str,
    *,
    backend: str,
    refinement: dict,
    events: list[dict],
) -> str:
    """Write one camera's report atomically; returns its path."""
    out = path(bag_id, chunk_id, cam_id)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Not NamedTemporaryFile: it creates 0600, which os.replace keeps, so a
    # reader running as another user couldn't open the report.
    tmp = out + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "schema_version": SCHEMA_VERSION,
                    "backend": backend,
                    "refinement": refinement,
                    "events": events,
                },
                fh,
                allow_nan=False,
            )
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    os.replace(tmp, out)
    return out


def read(bag_id: str, chunk_id: str, cam_id: str) -> dict:
    with open(path(bag_id, chunk_id, cam_id), encoding="utf-8") as fh:
        report = json.load(fh)
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path(bag_id, chunk_id, cam_id)}: mask_quality schema "
            f"{report.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    return report
