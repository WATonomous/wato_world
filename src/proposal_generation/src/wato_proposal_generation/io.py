"""Artifact I/O for proposal_generation. Reads upstream artifacts, writes own."""

from __future__ import annotations

from wato_common.storage import artifact_uri


def output_prefix(bag_id: str, chunk_id: str) -> str:
    return artifact_uri("proposal_generation", bag_id, chunk_id)
