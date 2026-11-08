"""Full non-ground scan availability, preserving original sweep point keys."""

from dataclasses import replace

import numpy as np
import pytest

from wato_common.artifacts import (
    ArtifactFormatError,
    lidar_sweeps,
    lifted_candidates,
    lifted_labels,
)
from wato_proposal_generation.frame_points import load

from ._staging import BAG, CHUNK, stage_chunk


@pytest.fixture(autouse=True)
def artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_ROOT_URI", f"file://{tmp_path}")


def test_full_unlabelled_sweep_is_available_and_ground_does_not_renumber_keys():
    stage_chunk()
    # Second scanner has no camera labels or gated motion source, but must
    # still supply candidates to seeds on the first scanner in this frame.
    lifted_labels.write_sweep(BAG, CHUNK, 1, lifted_labels.SweepLabels.empty())
    lifted_candidates.write_sweep(
        BAG, CHUNK, 1, lifted_candidates.SweepCandidates.empty()
    )
    sweeps = lidar_sweeps.ProcessedSweeps.load(BAG, CHUNK)
    world = sweeps.world(1)
    ground = np.zeros(world.n_points, bool)
    ground[120:130] = True
    lidar_sweeps.write_world(BAG, CHUNK, 1, replace(world, ground_mask=ground))
    frame = sweeps.frames("lidar_cc")[0]
    fp = load(BAG, CHUNK, frame, sweeps, np.array([], np.int64))
    second = fp.sweep_id == 1
    assert fp.point_index[second].tolist() == np.flatnonzero(~ground).tolist()
    assert np.all(fp.instance_id[second] == "")
    assert np.all(fp.cluster_id[second] == -1)
    assert np.all(fp.confidence[second] == 0)
    assert len(set(zip(fp.sweep_id, fp.point_index))) == fp.n
    # D failed the motion gate and has no lifted label on either sweep.
    assert np.any((fp.point_index >= 130) & (fp.point_index < 160))


def test_missing_ground_mask_stops_instead_of_clustering_the_road():
    stage_chunk()
    sweeps = lidar_sweeps.ProcessedSweeps.load(BAG, CHUNK)
    lidar_sweeps.write_world(BAG, CHUNK, 0, replace(sweeps.world(0), ground_mask=None))
    with pytest.raises(ArtifactFormatError, match="ground_mask.*Patchwork"):
        load(BAG, CHUNK, sweeps.frames("lidar_cc")[0], sweeps, np.array([]))


def test_competing_seeds_keep_one_frame_row_per_return_across_scanners():
    stage_chunk()
    for sid in (0, 1):
        lifted_candidates.write_sweep(
            BAG,
            CHUNK,
            sid,
            lifted_candidates.SweepCandidates(
                np.array([2, 2, 5], np.int32),
                np.array(["mA", "loser", "loser"]),
                np.array([0.9, 0.3, 0.4], np.float32),
            ),
        )
    sweeps = lidar_sweeps.ProcessedSweeps.load(BAG, CHUNK)
    fp = load(BAG, CHUNK, sweeps.frames("lidar_cc")[0], sweeps, np.array([]))
    seeds = fp.seeds()
    shared = np.intersect1d(seeds["mA"].point_indices, seeds["loser"].point_indices)
    assert list(zip(fp.sweep_id[shared], fp.point_index[shared])) == [(0, 2), (1, 2)]
    assert seeds["loser"].confidences.tolist() == pytest.approx([0.3, 0.4] * 2)
    assert len(set(zip(fp.sweep_id, fp.point_index))) == fp.n


def test_owned_labels_cannot_silently_substitute_for_missing_candidates():
    import os

    from wato_common.artifact_store import lifted_candidates_path, local_path

    stage_chunk()
    os.remove(local_path(lifted_candidates_path(BAG, CHUNK, 0)))
    sweeps = lidar_sweeps.ProcessedSweeps.load(BAG, CHUNK)
    with pytest.raises(ArtifactFormatError, match="re-run semantic_lifting"):
        load(BAG, CHUNK, sweeps.frames("lidar_cc")[0], sweeps, np.array([]))
