"""lidar_preprocessing pipeline orchestrator (stub)."""

from __future__ import annotations

from wato_lidar_preprocessing.config import ComponentConfig


def run(cfg: ComponentConfig, *, bag_id: str, chunk_id: str | None = None) -> None:
    """Process one bag (or one chunk) end-to-end and write artifacts.

    Stub implementation — fill in per the architecture doc when this component
    is built out.  See README in src/lidar_preprocessing/ for the expected contract.
    """
    raise NotImplementedError(
        f"lidar_preprocessing not implemented yet; bag={bag_id} chunk={chunk_id}"
    )
