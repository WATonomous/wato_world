"""open_vocab_discovery pipeline orchestrator (stub)."""

from __future__ import annotations

from wato_open_vocab_discovery.config import ComponentConfig


def run(cfg: ComponentConfig, *, bag_id: str, chunk_id: str | None = None) -> None:
    """Process one bag (or one chunk) end-to-end and write artifacts.

    Stub implementation — fill in per the architecture doc when this component
    is built out.  See README in src/open_vocab_discovery/ for the expected contract.
    """
    raise NotImplementedError(
        f"open_vocab_discovery not implemented yet; bag={bag_id} chunk={chunk_id}"
    )
