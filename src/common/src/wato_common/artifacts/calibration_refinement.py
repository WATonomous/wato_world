"""Camera refinement artifacts and explicit recorded/validated selection.

The completion manifest hashes candidate.json, effective.json and report.json.
Inputs use artifact-relative names so host/container relocation is supported.
Validated readers verify completion, input identity and each requested camera's
decision. A copied effective.json without its evidence is not validated.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from wato_common.artifact_store import (
    bag_root,
    calibration_path,
    calibration_refinement_dir,
    chunk_root,
    local_path,
)
from wato_common.artifacts import ArtifactFormatError
from wato_common.artifacts.calibration import Calibration
from wato_common.storage import artifact_root

Source = Literal["recorded", "validated"]
ACCEPTED = frozenset({"refined", "recorded_valid"})
OUTCOMES = ACCEPTED | {"insufficient_evidence", "rejected", "error", "not_requested"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: str | Path) -> dict:
    """Content identity, relative to the artifact tree whenever possible."""
    path = Path(path).resolve()
    root = Path(local_path(artifact_root())).resolve()
    try:
        location = {"relative_path": str(path.relative_to(root))}
    except ValueError:
        location = {"path": str(path)}
    stat = path.stat()
    # Sweeps/images used by calibration are individually small; hash the exact
    # selected inputs, not an entire recording or all unused sweeps.
    return {**location, "size_bytes": stat.st_size, "sha256": sha256(path)}


def fingerprint_path(entry: dict) -> Path:
    if "relative_path" in entry:
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid artifact-relative input path")
        return Path(local_path(artifact_root())) / relative
    return Path(entry["path"])


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def directory(bag_id: str) -> Path:
    return Path(local_path(calibration_refinement_dir(bag_id)))


def write(
    bag_id: str,
    candidate: Calibration,
    effective: Calibration,
    report: dict,
    inputs: list[dict],
    metadata: dict,
) -> Path:
    """Publish under the producer's bag lock; manifest is the commit record."""
    root = directory(bag_id)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)
    values = {
        "candidate.json": candidate.to_dict(),
        "effective.json": effective.to_dict(),
        "report.json": report,
    }
    outputs = {}
    for name, value in values.items():
        atomic_json(root / name, value)
        outputs[name] = sha256(root / name)
    atomic_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "bag_id": bag_id,
            "baseline_sha256": sha256(local_path(calibration_path(bag_id))),
            "inputs": inputs,
            "outputs": outputs,
            "metadata": metadata,
        },
    )
    return root


def read(bag_id: str, *, verify_inputs: bool = True) -> tuple[Calibration, dict, dict]:
    root = directory(bag_id)
    try:
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest["schema_version"] != 1 or manifest["bag_id"] != bag_id:
            raise ValueError("manifest version/bag mismatch")
        if sha256(local_path(calibration_path(bag_id))) != manifest["baseline_sha256"]:
            raise ValueError("recorded calibration changed")
        for name in ("candidate.json", "effective.json", "report.json"):
            if sha256(root / name) != manifest["outputs"][name]:
                raise ValueError(f"{name} changed or publication is incomplete")
        if verify_inputs:
            for entry in manifest["inputs"]:
                if fingerprint(fingerprint_path(entry)) != entry:
                    raise ValueError(f"input changed: {fingerprint_path(entry)}")
        report = json.loads((root / "report.json").read_text())
        if report["schema_version"] != 1 or report["bag_id"] != bag_id:
            raise ValueError("report version/bag mismatch")
        effective = Calibration.from_dict(
            json.loads((root / "effective.json").read_text()), source=str(root)
        )
        if set(report["cameras"]) != set(effective.cameras):
            raise ValueError("report/effective camera mismatch")
        for cid, result in report["cameras"].items():
            if result["status"] not in OUTCOMES:
                raise ValueError(f"invalid decision for {cid}")
        recorded = Calibration.load(bag_id).to_dict()
        selected = effective.to_dict()
        for key in ("ego_frame", "lidars", "static_transforms"):
            if recorded[key] != selected[key]:
                raise ValueError(f"camera refinement changed fixed {key}")
        if set(recorded["cameras"]) != set(selected["cameras"]):
            raise ValueError("camera refinement changed the sensor set")
        for cid in recorded["cameras"]:
            before = dict(recorded["cameras"][cid])
            after = dict(selected["cameras"][cid])
            old_transform = before.pop("ego_T_cam")
            new_transform = after.pop("ego_T_cam")
            if before != after:
                raise ValueError(
                    f"camera refinement changed fixed image geometry for {cid}"
                )
            if (
                report["cameras"][cid]["status"] != "refined"
                and old_transform != new_transform
            ):
                raise ValueError(f"unaccepted camera transform changed for {cid}")
            _ = effective.camera(cid).model
        if (root / "manifest.json").read_bytes() != manifest_bytes:
            raise ValueError("calibration publication changed while reading")
        return effective, report, manifest
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ArtifactFormatError(
            str(root),
            f"camera calibration missing, incomplete or stale: {exc}; "
            "rerun camera_calibration --force",
        ) from exc


@dataclass(frozen=True)
class ResolvedCalibration:
    calibration: Calibration
    source: Source
    path: str
    digest: str
    report_path: str | None = None
    report_digest: str | None = None

    def identity(self) -> dict:
        return {
            "source": self.source,
            "sha256": self.digest,
            "report_sha256": self.report_digest,
        }


def resolve(
    bag_id: str,
    source: Source = "recorded",
    required_cameras: Iterable[str] | None = None,
) -> ResolvedCalibration:
    if source == "recorded":
        path = local_path(calibration_path(bag_id))
        calibration = Calibration.load(bag_id)
        result = ResolvedCalibration(calibration, source, path, sha256(path))
    elif source == "validated":
        calibration, report, _ = read(bag_id)
        if not report.get("validation_qualified", False):
            raise ValueError("calibration validation profile is not qualified for use")
        cameras = set(
            calibration.cameras if required_cameras is None else required_cameras
        )
        failures = {
            cid: report["cameras"].get(cid, {}).get("status", "missing")
            for cid in cameras
            if report["cameras"].get(cid, {}).get("status") not in ACCEPTED
        }
        if failures:
            raise ValueError(f"validated camera calibration unavailable: {failures}")
        root = directory(bag_id)
        result = ResolvedCalibration(
            calibration,
            source,
            str(root / "effective.json"),
            sha256(root / "effective.json"),
            str(root / "report.json"),
            sha256(root / "report.json"),
        )
    else:
        raise ValueError(f"unknown camera calibration source {source!r}")
    for cid in required_cameras or ():
        _ = result.calibration.camera(cid).model
    return result


def check_output_identity(path: str | Path, selection: ResolvedCalibration) -> None:
    """Refuse existing outputs made with another calibration, including legacy
    outputs without identity when validated calibration is requested.
    """
    path = Path(path)
    if not path.exists():
        return
    manifest = json.loads(path.read_text())
    old = manifest.get("camera_calibration")
    if old == selection.identity():
        return
    if old is None and selection.source == "recorded":
        # Legacy recorded runs identify calibration in their input manifest.
        recorded = manifest.get("inputs", {}).get("calibration", {})
        if recorded.get("sha256") in (None, selection.digest):
            return
    raise ValueError(
        f"{path}: outputs use a different camera calibration; rerun this stage "
        "and downstream stages with --force"
    )


def _binding_path(bag_id: str, chunk_id: str | None, stage: str) -> Path:
    if not stage.replace("_", "").isalnum():
        raise ValueError("invalid stage name")
    root = bag_root(bag_id) if chunk_id is None else chunk_root(bag_id, chunk_id)
    return Path(local_path(root)) / f"camera_calibration_{stage}.json"


def output_binding(bag_id: str, chunk_id: str, stage: str) -> dict | None:
    for cid in (chunk_id, None):
        path = _binding_path(bag_id, cid, stage)
        if path.exists():
            return json.loads(path.read_text())
    return None


def bind_outputs(
    bag_id: str,
    chunk_id: str | None,
    stage: str,
    selection: ResolvedCalibration,
    *,
    force: bool,
    existing_outputs: bool,
) -> None:
    """Bind resumable/cached work before any image/depth/model processing."""
    binding = {
        "identity": selection.identity(),
        "calibration_path": Path(selection.path).as_uri(),
        "report_path": Path(selection.report_path).as_uri()
        if selection.report_path
        else None,
    }
    _bind(bag_id, chunk_id, stage, binding, force, existing_outputs)


def _bind(bag_id, chunk_id, stage, binding, force, existing_outputs):
    path = _binding_path(bag_id, chunk_id, stage)
    previous = json.loads(path.read_text()) if path.exists() else None
    if not force and existing_outputs:
        if previous is not None and previous.get("needs_force", False):
            raise ValueError(
                f"{stage}: a forced replacement did not complete; rerun with --force"
            )
        if previous is not None and previous["identity"] != binding["identity"]:
            raise ValueError(
                f"{stage}: cached outputs use different camera calibration; rerun {stage} and downstream with --force"
            )
        if previous is None and binding["identity"]["source"] == "validated":
            raise ValueError(
                f"{stage}: legacy outputs have no validated calibration identity; rerun with --force"
            )
        if previous is None and chunk_id is not None:
            old_manifest = (
                Path(local_path(chunk_root(bag_id, chunk_id)))
                / f"manifest_{stage}.json"
            )
            if old_manifest.exists():
                prior = json.loads(old_manifest.read_text())
                old_digest = (
                    prior.get("inputs", {}).get("calibration", {}).get("sha256")
                )
                if (
                    old_digest is not None
                    and old_digest != binding["identity"]["sha256"]
                ):
                    raise ValueError(
                        f"{stage}: recorded calibration changed; rerun with --force"
                    )
    # A failed forced switch must not bless old completed outputs with a new
    # identity. Fresh same-source partial caches remain resumable.
    atomic_json(path, {**binding, "needs_force": bool(force and existing_outputs)})


def finish_output_binding(bag_id: str, chunk_id: str, stage: str) -> None:
    for cid in (chunk_id, None):
        path = _binding_path(bag_id, cid, stage)
        if path.exists():
            binding = json.loads(path.read_text())
            atomic_json(path, {**binding, "needs_force": False})
            return


def inherited_identity(paths: Iterable[str | Path]) -> dict | None:
    """Camera identity is carried through proposal/tracking artifacts too."""
    identities = []
    missing = False
    for uri in paths:
        path = Path(local_path(str(uri)))
        data = json.loads(path.read_text()) if path.exists() else {}
        identity = data.get("camera_calibration")
        if identity is not None:
            identities.append(identity)
        else:
            missing = True
    if not identities:
        return None  # Historical recorded outputs retain their old behavior.
    if any(i != identities[0] for i in identities):
        raise ValueError(
            "upstream outputs mix different camera calibrations; rerun upstream stages with --force"
        )
    if missing and identities[0]["source"] == "validated":
        raise ValueError(
            "upstream outputs lack a consistent validated calibration identity"
        )
    return identities[0]


def bind_inherited_outputs(bag_id, chunk_id, stage, paths, *, force, existing_outputs):
    identity = inherited_identity(paths)
    if identity is not None:
        _bind(bag_id, chunk_id, stage, {"identity": identity}, force, existing_outputs)


def bind_from_bag_stage(bag_id, stage, upstream_stage, *, force, existing_outputs):
    """Bag-level consumers need no artificial chunk-index dependency."""
    upstream = output_binding(bag_id, "", upstream_stage)
    if upstream:
        _bind(bag_id, None, stage, upstream, force, existing_outputs)
        return upstream["identity"]
    return None


def require_upstream(selection: ResolvedCalibration, paths: Iterable[str | Path]):
    identity = inherited_identity(paths)
    if identity is not None and identity != selection.identity():
        raise ValueError(
            "upstream outputs use different camera calibration; rerun upstream stages with --force"
        )
    if identity is None and selection.source == "validated":
        raise ValueError(
            "upstream outputs have no validated calibration identity; rerun upstream stages"
        )
