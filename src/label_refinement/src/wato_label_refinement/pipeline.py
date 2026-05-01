"""label_refinement pipeline orchestrator (stub)."""

from __future__ import annotations

from wato_label_refinement.config import ComponentConfig


def run(cfg: ComponentConfig, *, bag_id: str, chunk_id: str | None = None) -> None:
    """Process one bag (or one chunk) end-to-end and write artifacts.

    Stub implementation — fill in per the architecture doc when this component
    is built out.  See README in src/label_refinement/ for the expected contract.
    """
    raise NotImplementedError(
        f"label_refinement not implemented yet; bag={bag_id} chunk={chunk_id}"
    )
