"""Step A2 — MapMOS inference (per-point moving/static logits).

Public surface kept narrow: orchestration and the result dataclass. The
real inference implementation lives in `inference.py`; today it ships as
a zero-prior stub so Step 1 of the integration plan (artifact wiring +
regression gate) can land independently of the MinkowskiEngine docker
stage.
"""

from .pipeline import MapMOSResult, process_chunk

__all__ = ["process_chunk", "MapMOSResult"]
