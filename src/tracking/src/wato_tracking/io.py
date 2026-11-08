"""Artifact I/O helpers for tracking.

Reads proposals through wato_common.artifacts.proposals and writes tracks
through wato_common.artifacts.tracks. Read another component's artifact only
through its wato_common.artifacts module, which owns the format.
"""

from __future__ import annotations

from wato_common.artifact_store import chunks_index_path
from wato_common.io.parquet_io import read_rows


def load_chunks(bag_id: str) -> list[dict]:
    return read_rows(chunks_index_path(bag_id))
