"""History-window helpers for MapMOS inference.

`get_past_rows` builds the list of previous sweeps that contextualize a
query sweep. The list filters out `valid=False` rows (plan non-negotiable
#17 — the network must never see a hole in the history window) and falls
back into the previous chunk's tail when the current chunk runs short
(plan: read-only prev-chunk loading).

`RollingHistoryCache` avoids reloading the same world NPZ N times as the
query window slides forward through the chunk. It is NOT thread-safe;
sweep processing is sequential within a chunk today.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import numpy as np

from wato_common.artifact_store import lidar_proc_index_path, local_path
from wato_common.io.parquet_io import read_rows
from wato_lidar_preprocessing.classify.io_helpers import (
    cache_byte_budget,
    load_world_full,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Past-sweep selection
# ---------------------------------------------------------------------------
def load_prev_chunk_meta_rows(bag_id: str, prev_chunk_id: str | None) -> list[dict] | None:
    """Read the previous chunk's lidar_proc_index, or None if unavailable.

    Three reasons to return None:
      1. There is no previous chunk (the first chunk of the bag).
      2. The previous chunk's index file doesn't exist (classify failed
         on the prior chunk, leaving no index behind).
      3. prev_chunk_id was explicitly passed as None.
    """
    if prev_chunk_id is None:
        return None
    prev_index_uri = lidar_proc_index_path(bag_id, prev_chunk_id)
    if not os.path.exists(local_path(prev_index_uri)):
        log.debug(
            "previous chunk %s index not found — first sweeps will have short history",
            prev_chunk_id,
        )
        return None
    return read_rows(prev_index_uri)


def get_past_rows(
    meta_rows: list[dict],
    sweep_id: int,
    n_past: int,
    prev_meta_rows: list[dict] | None = None,
) -> list[dict]:
    """Up to `n_past` valid rows preceding `sweep_id`, newest first.

    Filters out `valid=False` rows so the network never sees a hole in
    the history window (plan non-negotiable #17). Pads with the previous
    chunk's tail when the current chunk runs short.
    """
    try:
        current_idx = next(
            i for i, r in enumerate(meta_rows) if int(r["sweep_id"]) == sweep_id
        )
    except StopIteration as exc:
        raise ValueError(
            f"sweep_id {sweep_id!r} not present in current chunk meta_rows"
        ) from exc

    past = [
        r for r in meta_rows[:current_idx][::-1] if r.get("valid") is not False
    ][:n_past]

    if len(past) < n_past and prev_meta_rows:
        needed = n_past - len(past)
        tail = [
            r for r in prev_meta_rows[::-1] if r.get("valid") is not False
        ][:needed]
        past += tail

    if len(past) < n_past:
        log.debug(
            "sweep %d: only %d/%d past sweeps available", sweep_id, len(past), n_past
        )

    return past


# ---------------------------------------------------------------------------
# Rolling NPZ cache
# ---------------------------------------------------------------------------
class RollingHistoryCache:
    """LRU cache for `load_world_full(world_path)` results.

    NOT thread-safe. Sweep processing is sequential within a chunk today.
    If a future PR parallelizes sweeps inside a chunk, this is the first
    place a race will appear — gate access on an external lock or shard
    the cache per worker.

    Reuses the `WATO_LIDAR_CACHE_BYTES` budget from the classify cache so
    a single env var caps total in-memory NPZ footprint for the component.
    """

    def __init__(self, byte_budget: int | None = None):
        self._budget = byte_budget if byte_budget is not None else cache_byte_budget()
        self._entries: "OrderedDict[str, tuple]" = OrderedDict()
        self._bytes = 0

    def get_or_load(self, world_path_uri: str) -> tuple:
        """Returns the load_world_full tuple, hitting cache when possible."""
        if world_path_uri in self._entries:
            self._entries.move_to_end(world_path_uri)
            return self._entries[world_path_uri]
        loaded = load_world_full(world_path_uri)
        size = _approx_bytes(loaded)
        self._entries[world_path_uri] = loaded
        self._bytes += size
        self._evict_to_budget()
        return loaded

    def _evict_to_budget(self) -> None:
        while self._bytes > self._budget and len(self._entries) > 1:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= _approx_bytes(evicted)


def _approx_bytes(loaded_tuple: tuple) -> int:
    """Sum nbytes across the (xyz, intensity, origin, ground_mask) tuple."""
    total = 0
    for arr in loaded_tuple:
        if isinstance(arr, np.ndarray):
            total += int(arr.nbytes)
    return total
