"""Pinned model revisions for perception_2d.

Single source of truth for *which exact weights* this component runs. Imported
by both sides so they cannot drift apart:

  - ``scripts/fetch_models.py`` (host-side) — downloads these revisions into
    ``${MODELS_ROOT}``.
  - the runtime loaders — ``hf_revision()`` for detector / discovery / depth,
    ``torch_hub_repo()`` for embeddings — request the same revisions, so a
    pinned fetch and an unpinned load can't disagree.

WHY EVERY ENTRY CARRIES A REVISION
----------------------------------
This pipeline's output is training data. A HuggingFace repo's ``main`` branch
and a GitHub repo's default branch both move: ``IDEA-Research/grounding-dino-base``
can be re-uploaded, ``facebookresearch/dinov2`` can be force-pushed. Without a
revision, re-running ``fetch_models.py`` months apart populates ``MODELS_ROOT``
with different weights, the pipeline emits different labels, and nothing in the
artifact tree records that anything changed.

The revisions below are the ones present in ``data/models`` as of 2026-09-21 —
i.e. the weights that produced the labels in the tree today, not merely whatever
was current upstream.

TO UPGRADE A MODEL
------------------
Change the revision here, re-run ``fetch_models.py``, re-run the affected
stages. The new revision is recorded in every manifest written afterwards (see
``wato_common.provenance``), so old and new labels stay distinguishable.
"""

from __future__ import annotations

from typing import NamedTuple


class HFModel(NamedTuple):
    """A HuggingFace repo snapshot-downloaded into the HF_HOME cache."""

    repo_id: str
    revision: str
    note: str


class RawCheckpoint(NamedTuple):
    """A single file pulled out of an HF repo to a plain path on disk."""

    repo_id: str
    filename: str
    revision: str
    note: str


class TorchHubModel(NamedTuple):
    """A torch.hub entrypoint, pinned to a git ref rather than a branch."""

    repo: str
    entrypoint: str
    ref: str
    note: str


# Loaded at runtime via transformers / huggingface_hub ``from_pretrained``,
# which read the HF cache layout under HF_HOME.
HF_MODELS: dict[str, HFModel] = {
    "grounding_dino": HFModel(
        repo_id="IDEA-Research/grounding-dino-base",
        revision="12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        note="detector.py — AutoModelForZeroShotObjectDetection",
    ),
    "depth_anything_v2": HFModel(
        repo_id="depth-anything/Depth-Anything-V2-Large",
        revision="cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9",
        note="depth.py — metric depth via the DPT head",
    ),
    # Optional open-vocabulary discovery backend. Referenced by
    # config/perception_2d.yaml (discovery.model) and models/discovery.py, but
    # it was missing from this registry until 2026-09-21 — so it was never
    # pre-fetched. With HF_HUB_OFFLINE=1 and /data/models mounted read-only,
    # enabling discovery on a clean host would fail at load time. Listed here
    # so `fetch_models.py` pulls it like everything else.
    "florence_2": HFModel(
        repo_id="microsoft/Florence-2-large-ft",
        revision="4a12a2b54b7016a48a22037fbd62da90cd566f2a",
        note="discovery.py — dense-region captioning (optional vocab source)",
    ),
}

# Downloaded directly into MODELS_ROOT (NOT the HF cache): the runtime loader
# takes a plain filesystem path. Lands at ${MODELS_ROOT}/<filename>.
RAW_CHECKPOINTS: dict[str, RawCheckpoint] = {
    "sam2": RawCheckpoint(
        repo_id="facebook/sam2.1-hiera-large",
        filename="sam2.1_hiera_large.pt",
        revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
        note="sam2_tracker.py — build_sam2_video_predictor(ckpt_path); "
        "the hydra config ships inside the pinned `sam2` package",
    ),
}

# torch.hub models. The ref is appended as `repo:ref`, which also decides the
# TORCH_HOME cache directory name (facebookresearch_dinov2_<ref>) — so the
# runtime loader MUST use the same ref or it will miss the pre-fetched cache
# and try to reach the network.
TORCH_HUB_MODELS: dict[str, TorchHubModel] = {
    "dinov2": TorchHubModel(
        repo="facebookresearch/dinov2",
        entrypoint="dinov2_vitl14",
        ref="7764ea0f912e53c92e82eb78a2a1631e92725fc8",
        note="embeddings.py — DINOv2 ViT-L/14 ReID features",
    ),
}

ALL_TAGS: list[str] = [*HF_MODELS, *RAW_CHECKPOINTS, *TORCH_HUB_MODELS]


def torch_hub_repo(tag: str = "dinov2") -> str:
    """``repo:ref`` string for ``torch.hub.load`` — pinned, never a branch."""
    m = TORCH_HUB_MODELS[tag]
    return f"{m.repo}:{m.ref}"


def hf_revision(repo_id: str) -> str:
    """Pinned commit for an HF repo — what every ``from_pretrained`` must pass.

    Passing the SHA is required, not just tidy. ``fetch_models.py`` downloads
    by SHA, and huggingface_hub only writes ``refs/main`` when the requested
    revision is *not* already a commit hash — so a clean-host fetch leaves no
    ``main`` ref. A loader that omits ``revision=`` asks for ``main``, and under
    ``HF_HUB_OFFLINE=1`` cannot resolve it. Online, it would silently load
    whatever upstream ``main`` is today, making the manifest's recorded
    revision a lie.

    Raises KeyError for an unregistered repo: an unpinned model can't be
    pre-fetched or traced, so it must not load.
    """
    for m in HF_MODELS.values():
        if m.repo_id == repo_id:
            return m.revision
    raise KeyError(
        f"HF model {repo_id!r} is not in model_registry.HF_MODELS. Register it "
        "with a pinned commit SHA and re-run scripts/fetch_models.py."
    )


def revisions() -> dict[str, str]:
    """Flat ``{tag: revision}`` map for recording in an artifact manifest.

    This is what makes a label traceable: given a manifest you can name the
    exact weights that produced it, and given two manifests you can tell
    whether a difference in output is explained by a model change.
    """
    out: dict[str, str] = {}
    for tag, m in HF_MODELS.items():
        out[tag] = f"{m.repo_id}@{m.revision}"
    for tag, c in RAW_CHECKPOINTS.items():
        out[tag] = f"{c.repo_id}@{c.revision}"
    for tag, t in TORCH_HUB_MODELS.items():
        out[tag] = f"{t.repo}@{t.ref}"
    return out
