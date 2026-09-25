"""Tests for the pinned model registry.

The registry is the single source of truth for which weights perception_2d
runs. These tests exist to make it hard to *accidentally* unpin something — a
floating revision here silently changes every 2D label the pipeline emits.
"""

from __future__ import annotations

import re

import pytest

from wato_perception_2d import model_registry as reg

#: HF and GitHub both use 40-hex-char commit SHAs. A branch name like "main"
#: or a tag like "v1.0" fails this, which is the point.
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@pytest.mark.parametrize("tag", list(reg.HF_MODELS))
def test_hf_models_pin_an_exact_revision(tag):
    assert SHA_RE.match(reg.HF_MODELS[tag].revision), (
        f"{tag} must pin a full commit SHA, not a branch or tag — "
        "a moving revision silently changes the labeler"
    )


@pytest.mark.parametrize("tag", list(reg.RAW_CHECKPOINTS))
def test_raw_checkpoints_pin_an_exact_revision(tag):
    assert SHA_RE.match(reg.RAW_CHECKPOINTS[tag].revision)


@pytest.mark.parametrize("tag", list(reg.TORCH_HUB_MODELS))
def test_torch_hub_models_pin_a_commit_not_a_branch(tag):
    assert SHA_RE.match(reg.TORCH_HUB_MODELS[tag].ref)


def test_torch_hub_repo_is_ref_qualified():
    """torch.hub caches by ref, so the runtime loader must ask for repo:<sha>.

    A bare "facebookresearch/dinov2" resolves to the default branch and looks
    for a differently-named cache directory, missing the pre-fetched weights.
    """
    repo = reg.torch_hub_repo("dinov2")
    assert ":" in repo
    owner_repo, ref = repo.rsplit(":", 1)
    assert owner_repo == "facebookresearch/dinov2"
    assert SHA_RE.match(ref)


def test_revisions_covers_every_registered_model():
    """Anything added to the registry must show up in the manifest block."""
    revs = reg.revisions()
    assert set(revs) == set(reg.ALL_TAGS)
    for tag, value in revs.items():
        assert "@" in value, f"{tag} revision string must be repo@sha"


def test_config_referenced_models_are_registered():
    """Models named in perception_2d.yaml must be pre-fetchable.

    The container runs with HF_HUB_OFFLINE=1 and /data/models mounted
    read-only, so a model that is referenced but not in the registry is never
    downloaded by fetch_models.py and fails at load time on a clean host.
    Florence-2 was exactly this case before 2026-09-21.
    """
    registered = {m.repo_id for m in reg.HF_MODELS.values()}
    assert "microsoft/Florence-2-large-ft" in registered
    assert "IDEA-Research/grounding-dino-base" in registered


@pytest.mark.parametrize("tag", list(reg.HF_MODELS))
def test_hf_revision_returns_the_pinned_sha(tag):
    m = reg.HF_MODELS[tag]
    assert reg.hf_revision(m.repo_id) == m.revision


def test_hf_revision_refuses_unregistered_repo():
    """An unregistered model has no pin and no pre-fetch — it must not load."""
    with pytest.raises(KeyError, match="not in model_registry"):
        reg.hf_revision("someone/unpinned-model")


def test_hf_loaders_pass_the_pinned_revision(monkeypatch):
    """Every HF load must request the registry SHA, never the default "main".

    fetch_models.py downloads by SHA, which writes no refs/main, so an
    offline from_pretrained(repo_id) with no revision can't resolve on a clean
    host. Captures the kwargs each loader hands to transformers.
    """
    import sys
    import types

    calls: list[tuple[str, dict]] = []

    class _Fake:
        @classmethod
        def from_pretrained(cls, repo_id, **kw):
            calls.append((repo_id, kw))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoProcessor = _Fake
    fake_tf.AutoModelForZeroShotObjectDetection = _Fake
    fake_tf.AutoModelForCausalLM = _Fake
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    from wato_perception_2d.models.detector import GroundingDinoDetector
    from wato_perception_2d.models.discovery import Florence2Discovery

    for cls, repo in (
        (GroundingDinoDetector, "IDEA-Research/grounding-dino-base"),
        (Florence2Discovery, "microsoft/Florence-2-large-ft"),
    ):
        calls.clear()
        obj = cls.__new__(cls)
        obj._model, obj._model_id, obj._device = None, repo, "cpu"
        obj._load()
        assert calls, f"{cls.__name__} made no from_pretrained call"
        for repo_id, kw in calls:
            assert kw.get("revision") == reg.hf_revision(
                repo_id
            ), f"{cls.__name__} loaded {repo_id} without the pinned revision"
