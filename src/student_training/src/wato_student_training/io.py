"""Artifact I/O for student_training. Reads upstream artifacts, writes own."""

from __future__ import annotations

from wato_common.storage import artifact_uri


def output_prefix(bag_id: str, chunk_id: str) -> str:
    return artifact_uri("student_training", bag_id, chunk_id)
