"""One module per artifact that crosses a component boundary.

Each module owns its artifact's path recipe, its on-disk encoding, and both
halves of the contract: the producer calls the write half, every consumer
calls the read half, and the two are round-trip tested together. No other code
builds these filenames or opens these files. Consumer images install only
src/common and their own package, so a reader in the producer's package is
unreachable to them — that is why both halves live here.

Reads return typed records and check the invariants the writer guarantees
(required keys, array lengths), raising ArtifactFormatError instead of
defaulting a missing value.
"""

from __future__ import annotations


class ArtifactFormatError(ValueError):
    """An artifact on disk doesn't match the format its module writes."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason
