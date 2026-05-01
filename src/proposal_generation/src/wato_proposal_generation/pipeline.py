"""proposal_generation pipeline orchestrator (stub)."""

from __future__ import annotations

from wato_proposal_generation.config import ComponentConfig


def run(cfg: ComponentConfig, *, bag_id: str, chunk_id: str | None = None) -> None:
    """Process one bag (or one chunk) end-to-end and write artifacts.

    Stub implementation — fill in per the architecture doc when this component
    is built out.  See README in src/proposal_generation/ for the expected contract.
    """
    raise NotImplementedError(
        f"proposal_generation not implemented yet; bag={bag_id} chunk={chunk_id}"
    )
