"""Linking each chunk's tracks to the next chunk's across their shared span.

Every id upstream restarts per chunk, so an object crossing a chunk seam is
tracked once per chunk, with nothing in common but time. Both chunks decode
the same sweeps over the span they share, and the same object's box is
usually the same in both (docs/tracking_measurements.md), so a track of chunk
k and one of chunk k+1 that follow one object have their boxes in one place at
the instants both saw. They are linked on that agreement (SUSHI's shared-frame
cost), one to one by the Hungarian method, maximising the instants agreed on:
a short fragment that agrees everywhere must not take the link from the long
track of the same object. A chain of links is one object over the bag.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from wato_common.chunk_spans import SharedSpan
from wato_tracking.config import StitchConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkTrack:
    """One chunk's track: its id and its box centre at each instant it saw."""

    track_id: str
    chunk_id: str
    centres: dict[int, tuple[float, float]]  # reference_timestamp_ns -> (cx, cy)


def agreement(
    a: ChunkTrack, b: ChunkTrack, span: SharedSpan, max_dist_m: float
) -> tuple[float, int]:
    """(share, n): n is the instants of the span both saw where their centres
    are within max_dist_m; share is n over the instants the shorter of the two
    saw in the span."""
    in_a = {t for t in a.centres if t in span}
    in_b = {t for t in b.centres if t in span}
    if not in_a or not in_b:
        return 0.0, 0
    n = sum(math.dist(a.centres[t], b.centres[t]) <= max_dist_m for t in in_a & in_b)
    return n / min(len(in_a), len(in_b)), n


def links(
    earlier: list[ChunkTrack],
    later: list[ChunkTrack],
    span: SharedSpan,
    cfg: StitchConfig,
) -> list[tuple[int, int]]:
    """(i, j) pairs linking earlier[i] to later[j] across the span, one to
    one: of the pairs that agree enough (config `stitch`), those agreeing at
    the most instants in total."""
    rows = [i for i, t in enumerate(earlier) if any(x in span for x in t.centres)]
    cols = [j for j, t in enumerate(later) if any(x in span for x in t.centres)]
    if not rows or not cols:
        return []
    # Cost: minus the instants agreed on; 1 marks a pair that can't link.
    cost = np.ones((len(rows), len(cols)))
    for r, i in enumerate(rows):
        for c, j in enumerate(cols):
            share, n = agreement(earlier[i], later[j], span, cfg.max_centre_dist_m)
            if n >= cfg.min_agreeing_frames and share >= cfg.min_agreement:
                cost[r, c] = -n
    rs, cs = linear_sum_assignment(cost)
    out = [(rows[r], cols[c]) for r, c in zip(rs, cs) if cost[r, c] < 0]
    log.info(
        "seam %s|%s: %d and %d tracks in the shared span, %d linked",
        span.earlier,
        span.later,
        len(rows),
        len(cols),
        len(out),
    )
    return out


def chains(
    per_chunk: list[list[ChunkTrack]], spans: list[SharedSpan], cfg: StitchConfig
) -> list[list[ChunkTrack]]:
    """Every track once, in chains of linked tracks in chunk order.

    per_chunk is each chunk's tracks in chunk order; spans[k] is the span
    chunks k and k+1 share (wato_common.chunk_spans.shared_spans).
    """
    if len(spans) != max(len(per_chunk) - 1, 0):
        raise ValueError(f"{len(per_chunk)} chunks but {len(spans)} shared spans")
    nxt: dict[tuple[int, int], tuple[int, int]] = {}
    for k, span in enumerate(spans):
        for i, j in links(per_chunk[k], per_chunk[k + 1], span, cfg):
            nxt[(k, i)] = (k + 1, j)
    linked_to = set(nxt.values())
    out = []
    for k, ts in enumerate(per_chunk):
        for i in range(len(ts)):
            if (k, i) in linked_to:
                continue
            chain = [(k, i)]
            while chain[-1] in nxt:
                chain.append(nxt[chain[-1]])
            out.append([per_chunk[a][b] for a, b in chain])
    return out
