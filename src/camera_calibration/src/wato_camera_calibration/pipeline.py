"""Bag-level orchestration. Recorded calibration is always the fitting prior."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path


from wato_common.artifact_store import calibration_path, local_path
from wato_common.artifacts.calibration import Calibration
from wato_common.artifacts import calibration_refinement as artifact
from wato_common.provenance import collect as provenance

from wato_camera_calibration.config import ComponentConfig
from wato_camera_calibration.sampling import collect
from wato_camera_calibration.solver import backend_identity, prepare, run_fits
from wato_camera_calibration.validation import evaluate
from wato_camera_calibration.viz import overlay, write_index

log = logging.getLogger(__name__)


def implementation_identity():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(cfg: ComponentConfig, *, bag_id: str, cameras=None, force=False):
    if not cfg.enabled:
        log.info("camera calibration disabled; recorded calibration remains selected")
        return {"disabled": True}
    baseline = Calibration.load(bag_id)
    requested = sorted(set(cameras or cfg.cameras or baseline.cameras))
    if set(requested) - set(baseline.cameras):
        raise ValueError(f"unknown cameras: {set(requested) - set(baseline.cameras)}")
    for cid in requested:
        if Path(cid).name != cid or cid in (".", ".."):
            raise ValueError("camera IDs must be single path names")
        _ = baseline.camera(cid).model
    if cfg.sampling.lidar_id not in baseline.lidars:
        raise ValueError(f"unknown reference LiDAR {cfg.sampling.lidar_id!r}")
    root = artifact.directory(bag_id)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another calibration writer is running for this bag"
            ) from exc
        backend = (
            backend_identity(cfg.solver) if cfg.mode == "run" else {"mode": "prepare"}
        )
        settings = {
            "config": cfg.model_dump(),
            "cameras": requested,
            "backend": backend,
            "implementation_sha256": implementation_identity(),
        }
        if (root / "manifest.json").exists() and not force:
            _, report, manifest = artifact.read(bag_id)
            if manifest["metadata"].get("settings") != settings:
                raise ValueError(
                    "camera calibration settings/backend changed; rerun with --force"
                )
            log.info("camera calibration complete and inputs unchanged; skipping")
            return report
        annotations = {}
        sources = {Path(local_path(calibration_path(bag_id)))}
        if cfg.validation.correspondences:
            annotation_path = Path(cfg.validation.correspondences)
            data = json.loads(annotation_path.read_text())
            if data.get("schema_version") != 1 or data.get("bag_id") != bag_id:
                raise ValueError("independent correspondence version/bag mismatch")
            annotations = data["cameras"]
            sources.add(annotation_path)
        candidates, effective = dict(baseline.cameras), dict(baseline.cameras)
        report = {
            "schema_version": 1,
            "bag_id": bag_id,
            "validation_qualified": cfg.validation.qualified,
            "validation_profile_id": cfg.validation.profile_id,
            "cameras": {cid: {"status": "not_requested"} for cid in baseline.cameras},
        }
        for cid in requested:
            log.info("%s: collecting independent static views", cid)
            # Workspaces/overlays belong to this attempt. Mixing an older fit's
            # pictures into a new report would misrepresent its evidence.
            for owned in (root / "workspace" / cid, root / "diagnostics" / cid):
                if owned.exists():
                    shutil.rmtree(owned)
            cam = baseline.camera(cid)
            try:
                samples, used, rejections = collect(bag_id, cam, cfg.sampling)
                sources.update(used)
                pairs = prepare(samples, root / "workspace" / cid / "pairs")
                summary = {"sample_index": pairs, "sampling_rejections": rejections}
                counts = {
                    split: sum(s.split == split for s in samples)
                    for split in ("train", "validation", "test")
                }
                log.info("%s: %s views", cid, counts)
                blocks = {
                    split: len({s.block for s in samples if s.split == split})
                    for split in counts
                }
                if (
                    cfg.mode == "prepare"
                    or any(
                        n < cfg.validation.min_views_per_split for n in counts.values()
                    )
                    or any(
                        n < cfg.validation.min_blocks_per_split for n in blocks.values()
                    )
                    or blocks["train"] < cfg.solver.subsets
                ):
                    report["cameras"][cid] = {
                        **summary,
                        "status": "insufficient_evidence",
                        "views": counts,
                        "blocks": blocks,
                        "reasons": [
                            "prepare_only"
                            if cfg.mode == "prepare"
                            else "insufficient_split_views"
                        ],
                    }
                    continue
                candidate, transforms, fits = run_fits(
                    cam, pairs, cfg.solver, backend, root / "workspace" / cid / "fits"
                )
                candidates[cid] = replace(cam, ego_T_cam=candidate)
                decision = evaluate(
                    samples,
                    cam,
                    candidate,
                    transforms,
                    cfg.validation,
                    cfg.solver,
                    annotations,
                )
                report["cameras"][cid] = {
                    **summary,
                    **decision,
                    "fits": fits,
                    "fit_transforms": [t.tolist() for t in transforms],
                }
                if decision["status"] == "refined":
                    effective[cid] = candidates[cid]
                overlay(samples, cam, candidate, root / "diagnostics" / cid)
                log.info("%s: %s (%s)", cid, decision["status"], decision["reasons"])
            except (
                ValueError,
                OSError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as exc:
                log.exception("%s: calibration failed", cid)
                report["cameras"][cid] = {"status": "error", "reasons": [str(exc)]}
        inputs = [artifact.fingerprint(p) for p in sorted(sources)]
        artifact.write(
            bag_id,
            replace(
                baseline,
                cameras=candidates,
                version=baseline.version + "__camera_candidate",
            ),
            replace(
                baseline,
                cameras=effective,
                version=baseline.version + "__camera_effective",
            ),
            report,
            inputs,
            {
                "settings": settings,
                "provenance": provenance(component="camera_calibration"),
            },
        )
        write_index(root, report)
        return report
