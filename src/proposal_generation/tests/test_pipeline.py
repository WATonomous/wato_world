"""End to end over one staged chunk (tests/_staging.py)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace

import numpy as np
import pytest

from wato_common import manifest as common_manifest
from wato_common.artifact_store import (
    chunk_root,
    local_path,
    motion_clusters_path,
    proposals_path,
    proposal_refinement_path,
)
from wato_common.artifacts import lifted_candidates, lifted_labels, proposals
from wato_common.artifacts.lidar_sweeps import CanonicalLidarError
from wato_common.io.parquet_io import read_rows
from wato_proposal_generation.config import ComponentConfig, FramesConfig, LiftedConfig
from wato_proposal_generation.pipeline import process_chunk, run

from ._staging import BAG, CHUNK, N_FRAMES, sweep_id, stage_chunk


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def _cfg(canonical: str | None = "lidar_cc") -> ComponentConfig:
    return ComponentConfig(
        frames=FramesConfig(canonical_lidar=canonical),
        lifted=LiftedConfig(support_method="scan_clusters"),
    )


def _by_id() -> dict[str, proposals.Proposal]:
    return {p.proposal_id: p for p in proposals.read(BAG, CHUNK)}


def test_every_object_gets_its_source_and_fused_rows():
    stage_chunk()
    counts = process_chunk(_cfg(), BAG, CHUNK)
    # Per frame: A and C (D fails the motion gate) / A and B / A+B+C fused.
    assert counts == {
        "lidar_mos": 2 * N_FRAMES,
        "lifted": 2 * N_FRAMES,
        "fused": 3 * N_FRAMES,
        "fused_matched": N_FRAMES,
    }
    rows = _by_id()
    for f in range(N_FRAMES):
        ref = sweep_id(f, "lidar_cc")
        frame_rows = [p for p in rows.values() if p.sweep_id == ref]
        assert {p.sweep_ids for p in frame_rows} == {(ref, ref + 1)}
        assert {p.frame_id for p in frame_rows} == {f}

        mos_a, mos_c = rows[f"mos-{3 * f}"], rows[f"mos-{3 * f + 1}"]
        assert mos_a.cls == "car" and mos_a.supporting_masklet_ids == ("mA",)
        assert mos_c.cls is None and mos_c.supporting_cam_ids == ()
        assert mos_a.motion_score == 2.0 and mos_a.lift_confidence is None
        # Step F's box started 0.3 m up; it now reaches the ground.
        assert mos_a.cz - mos_a.h / 2 == pytest.approx(0.0)

        lifted_a = rows[f"lifted-mA-{ref:06d}"]
        # The bleed point at (30, 30) is not part of car A's box.
        assert lifted_a.cx == pytest.approx(10.0, abs=0.5)
        assert lifted_a.lidar_point_count == 80
        assert lifted_a.lift_confidence == pytest.approx(0.9)

        fused = [p for p in frame_rows if p.provenance == "fused"]
        pair = [p for p in fused if len(p.member_proposal_ids) == 2]
        assert len(pair) == 1
        assert set(pair[0].member_proposal_ids) == {
            mos_a.proposal_id,
            lifted_a.proposal_id,
        }
        assert pair[0].cls == "car"
        alone = {p.member_proposal_ids[0]: p for p in fused if p not in pair}
        assert set(alone) == {mos_c.proposal_id, f"lifted-mB-{ref:06d}"}
        assert alone[mos_c.proposal_id].cls is None
        assert alone[mos_c.proposal_id].motion_score is None


def test_recovered_static_support_survives_io_boxing_and_unmatched_fusion():
    stage_chunk()
    for sid in range(2 * N_FRAMES):
        labels = lifted_labels.read_sweep(BAG, CHUNK, sid)
        # Remove ten of each scanner's forty B labels. The returns remain in
        # the full scan, without motion support. DBSCAN may leave peripheral
        # returns as noise; matching must still recover the missing surface.
        drop = np.flatnonzero(labels.instance_ids == "mB")[-10:]
        keep = ~np.isin(np.arange(labels.point_indices.size), drop)
        lifted_labels.write_sweep(
            BAG,
            CHUNK,
            sid,
            replace(
                labels,
                **{
                    name: getattr(labels, name)[keep]
                    for name in (
                        "point_indices",
                        "instance_ids",
                        "classes",
                        "confidences",
                        "n_supporting_cameras",
                        "n_disagreeing_cameras",
                    )
                },
            ),
        )
        candidates = lifted_candidates.read_sweep(BAG, CHUNK, sid)
        drop = np.flatnonzero(candidates.instance_ids == "mB")[-10:]
        keep = ~np.isin(np.arange(candidates.point_indices.size), drop)
        lifted_candidates.write_sweep(
            BAG,
            CHUNK,
            sid,
            replace(
                candidates,
                point_indices=candidates.point_indices[keep],
                instance_ids=candidates.instance_ids[keep],
                confidences=candidates.confidences[keep],
            ),
        )
    process_chunk(_cfg(), BAG, CHUNK)
    rows = _by_id()
    audit = read_rows(proposal_refinement_path(BAG, CHUNK))
    assert len(audit) == 2 * N_FRAMES
    for record in [r for r in audit if r["instance_id"] == "mB"]:
        sid = record["sweep_id"]
        source = rows[f"lifted-mB-{sid:06d}"]
        assert 78 <= source.lidar_point_count <= 80
        assert source.lift_seed_point_count == 60
        fused = next(
            p for p in rows.values() if p.member_proposal_ids == (source.proposal_id,)
        )
        assert fused.lidar_point_count == source.lidar_point_count
        assert fused.lift_seed_point_count is None
        assert record["status"] == "replaced" and record["emitted"]
        assert record["n_seed_points"] == 60
        assert record["n_refined_points"] == source.lidar_point_count
        assert record["n_added_points"] >= 18 and record["n_removed_points"] == 0
        assert record["best_iou"] == pytest.approx(60 / source.lidar_point_count)


def test_run_skips_done_chunks_and_writes_a_manifest(caplog):
    caplog.set_level(logging.INFO)
    stage_chunk()
    run(_cfg(), bag_id=BAG)
    manifest = os.path.join(
        local_path(chunk_root(BAG, CHUNK)),
        common_manifest.component_manifest_name("proposal_generation"),
    )
    with open(manifest, encoding="utf-8") as fh:
        assert "motion_clusters" in json.load(fh)["inputs"]
    mtime = os.path.getmtime(local_path(proposals_path(BAG, CHUNK)))
    run(_cfg(), bag_id=BAG)
    assert "already complete" in caplog.text
    assert os.path.getmtime(local_path(proposals_path(BAG, CHUNK))) == mtime
    run(_cfg(), bag_id=BAG, force=True)


def test_a_canonical_lidar_lidar_preprocessing_did_not_use_fails():
    """Every frame has one lidar_ne sweep too, but Step F stamped its clusters
    with lidar_cc's time."""
    stage_chunk()
    with pytest.raises(CanonicalLidarError, match="Step F cluster"):
        process_chunk(_cfg("lidar_ne"), BAG, CHUNK)
    with pytest.raises(CanonicalLidarError, match="lidar_nw"):
        process_chunk(_cfg("lidar_nw"), BAG, CHUNK)


def test_without_a_canonical_lidar_step_f_frames_disagree():
    """Frames regrouped per sweep no longer match Step F's per-tick clusters."""
    stage_chunk()
    with pytest.raises(CanonicalLidarError, match="Step F cluster"):
        process_chunk(_cfg(None), BAG, CHUNK)


def test_missing_upstream_names_the_step_to_run():
    stage_chunk()
    os.remove(local_path(motion_clusters_path(BAG, CHUNK)))
    with pytest.raises(FileNotFoundError, match="motion_clusters.parquet"):
        process_chunk(_cfg(), BAG, CHUNK)


def test_false_provisional_owner_cannot_erase_true_objects_candidate_support():
    stage_chunk()
    ghost = lifted_labels.LiftedInstance(
        "ghost", "truck", ("ghost",), ("CAM_FRONT",), 24, 4
    )
    lifted_labels.write_instances(
        BAG, CHUNK, lifted_labels.read_instances(BAG, CHUNK) + [ghost]
    )
    for sid in range(2 * N_FRAMES):
        labels = lifted_labels.read_sweep(BAG, CHUNK, sid)
        # Every real A return lost its provisional 2D ownership. The retained
        # independent mA hypothesis still claims all of them, plus one bleed.
        ids = labels.instance_ids.copy()
        classes = labels.classes.copy()
        ids[ids == "mA"] = "ghost"
        classes[ids == "ghost"] = "truck"
        lifted_labels.write_sweep(
            BAG, CHUNK, sid, replace(labels, instance_ids=ids, classes=classes)
        )
        candidates = lifted_candidates.read_sweep(BAG, CHUNK, sid)
        lifted_candidates.write_sweep(
            BAG,
            CHUNK,
            sid,
            lifted_candidates.SweepCandidates(
                np.r_[candidates.point_indices, np.arange(6)].astype(np.int32),
                np.r_[candidates.instance_ids, ["ghost"] * 6],
                np.r_[candidates.confidences, np.full(6, 1.0)].astype(np.float32),
                np.r_[candidates.frustum_point_indices, np.arange(6)].astype(np.int32),
                np.r_[candidates.frustum_instance_ids, ["ghost"] * 6],
            ),
        )
    cfg = _cfg()
    # No scan expansion can hide lost claims: the default radius path only
    # removes seed outliers, so all 80 A returns must come from candidates.
    cfg.lifted.support_method = "largest_group"
    process_chunk(cfg, BAG, CHUNK)
    rows = _by_id()
    for f in range(N_FRAMES):
        sid = sweep_id(f, "lidar_cc")
        real = rows[f"lifted-mA-{sid:06d}"]
        assert real.lidar_point_count == real.lift_seed_point_count == 80
        assert real.cls == "car"
        assert f"lifted-ghost-{sid:06d}" not in rows
        assert rows[f"mos-{3 * f}"].cls == "car"
        assert rows[f"mos-{3 * f}"].supporting_masklet_ids == ("mA",)
    audit = read_rows(proposal_refinement_path(BAG, CHUNK))
    assert all(not r["emitted"] for r in audit if r["instance_id"] == "ghost")


def _verifying(method: str = "largest_group") -> ComponentConfig:
    cfg = _cfg()
    cfg.lifted.support_method = method
    return cfg


@pytest.mark.parametrize("method", ["largest_group", "scan_clusters"])
def test_verified_lifted_rows_carry_the_verdict_and_fused_rows_do_not(method):
    """Cars A and B: every return is claimed and inside the frustum, so each
    is one distinct segment (IoU near 1, containment 1) whatever boxes it."""
    stage_chunk()
    process_chunk(_verifying(method), BAG, CHUNK)
    rows = proposals.read(BAG, CHUNK)
    lifted = [p for p in rows if p.provenance == "lifted"]
    assert lifted and all(p.lidar_verified is True for p in lifted)
    assert all(p.lidar_verified is None for p in rows if p.provenance != "lifted")
    audit = read_rows(proposal_refinement_path(BAG, CHUNK))
    assert {r["verify_status"] for r in audit} == {"verified"}
    assert all(r["verify_containment"] == 1.0 for r in audit)


def test_a_segment_reaching_beyond_the_frustum_is_not_verified():
    """Restrict car B's claims and frustum to 70% of its returns: the B
    segment still matches the seed (IoU 0.7 > 0.5) but 30% of it lies outside
    the frustum, so it belongs to something larger than the mask (FGR)."""
    stage_chunk()
    for f in range(N_FRAMES):
        for sid in (sweep_id(f, "lidar_cc"), sweep_id(f, "lidar_ne")):
            c = lifted_candidates.read_sweep(BAG, CHUNK, sid)
            b = np.flatnonzero(c.instance_ids == "mB")
            drop = b[: int(round(0.3 * b.size))]
            keep = np.setdiff1d(np.arange(c.point_indices.size), drop)
            lifted_candidates.write_sweep(
                BAG,
                CHUNK,
                sid,
                lifted_candidates.SweepCandidates(
                    c.point_indices[keep],
                    c.instance_ids[keep],
                    c.confidences[keep],
                    c.point_indices[keep],
                    c.instance_ids[keep],
                ),
            )
    process_chunk(_verifying(), BAG, CHUNK)
    audit = {
        r["instance_id"]: r for r in read_rows(proposal_refinement_path(BAG, CHUNK))
    }
    assert audit["mA"]["lidar_verified"] is True
    assert audit["mB"]["verify_status"] == "containment_below_threshold"
    assert audit["mB"]["verify_containment"] == pytest.approx(0.7, abs=0.05)
    rows = proposals.read(BAG, CHUNK)
    assert all(
        p.lidar_verified is False
        for p in rows
        if p.provenance == "lifted" and p.proposal_id.startswith("lifted-mB")
    )


def test_without_verification_the_verdict_is_null():
    stage_chunk()
    cfg = _cfg()
    cfg.lifted.verification.enabled = False
    process_chunk(cfg, BAG, CHUNK)
    rows = proposals.read(BAG, CHUNK)
    assert all(p.lidar_verified is None for p in rows)
    audit = read_rows(proposal_refinement_path(BAG, CHUNK))
    assert all(
        r["lidar_verified"] is None and r["verify_status"] is None for r in audit
    )


def _claim_fraction(instance: str, fraction: float, frustum_all: bool) -> None:
    """Keep only `fraction` of an instance's claims in every sweep; its
    frustum keeps all its returns (frustum_all) or only the kept claims."""
    for f in range(N_FRAMES):
        for sid in (sweep_id(f, "lidar_cc"), sweep_id(f, "lidar_ne")):
            c = lifted_candidates.read_sweep(BAG, CHUNK, sid)
            mine = np.flatnonzero(c.instance_ids == instance)
            drop = mine[: int(round((1 - fraction) * mine.size))]
            keep = np.setdiff1d(np.arange(c.point_indices.size), drop)
            frustum = np.ones(c.frustum_point_indices.size, bool)
            if not frustum_all:
                # Only this instance's frustum shrinks, to its kept claims.
                frustum &= ~(
                    (c.frustum_instance_ids == instance)
                    & np.isin(c.frustum_point_indices, c.point_indices[drop])
                )
            lifted_candidates.write_sweep(
                BAG,
                CHUNK,
                sid,
                lifted_candidates.SweepCandidates(
                    c.point_indices[keep],
                    c.instance_ids[keep],
                    c.confidences[keep],
                    c.frustum_point_indices[frustum],
                    c.frustum_instance_ids[frustum],
                ),
            )


def test_verified_clusters_box_the_verifying_segment_and_leave_the_rest():
    """mA claims 70% of car A, all of it inside its frustum: verified, so its
    box takes the whole A segment (80 returns per frame). mB claims 70% of B
    with a frustum of only those: unverified, so largest_group keeps its 70%."""
    stage_chunk()
    _claim_fraction("mA", 0.7, frustum_all=True)
    _claim_fraction("mB", 0.7, frustum_all=False)
    process_chunk(_verifying("verified_clusters"), BAG, CHUNK)
    rows = _by_id()
    audit = {
        (r["sweep_id"], r["instance_id"]): r
        for r in read_rows(proposal_refinement_path(BAG, CHUNK))
    }
    for f in range(N_FRAMES):
        ref = sweep_id(f, "lidar_cc")
        a, b = rows[f"lifted-mA-{ref:06d}"], rows[f"lifted-mB-{ref:06d}"]
        assert a.lidar_verified and a.lidar_point_count == 80
        assert a.lift_seed_point_count == 56  # 70% of 80 claimed returns
        assert audit[(ref, "mA")]["status"] == "verified_cluster"
        assert not b.lidar_verified and b.lidar_point_count == 56
        assert audit[(ref, "mB")]["status"] == "largest_group"


def test_verified_clusters_need_verification():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="verification"):
        LiftedConfig(
            support_method="verified_clusters",
            verification={"enabled": False},
        )
