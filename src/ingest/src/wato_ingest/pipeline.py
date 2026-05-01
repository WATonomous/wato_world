"""Backwards-compat thin wrapper.

The original skeleton exposed `run(cfg, bag_id, chunk_id)`.  The real
implementation lives in `runner.run_bag(...)`.  Keep this stub for any
external code that imports `wato_ingest.pipeline.run` directly.
"""

from __future__ import annotations

from wato_ingest.config import IngestConfig
from wato_ingest.runner import run_bag


def run(cfg: IngestConfig, *, bag_id: str, chunk_id: str | None = None) -> None:
    """Compatibility shim — prefer calling runner.run_bag directly."""
    raise NotImplementedError(
        "Ingest needs the bag's filesystem path; call runner.run_bag(...) "
        "or use `python -m wato_ingest run --bag <path>` instead."
    )


__all__ = ["run", "run_bag"]
