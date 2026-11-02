"""Compare this run's masklets with a baseline run's, to review what a change
to the tracking rules removed.

A rule change (an ego mask, rules that end tracks) should end junk tracks and
keep every real object. Both runs box the same images with the same detector,
so an object the change lost shows up as a baseline mask with no current mask
on it. ``compare`` samples every ``every``-th image of each camera, matches
each mask there to the other run's masks of the same image (pixel IoU >=
MATCH_IOU), and lists the baseline masklets with unmatched masks, largest loss
first, as picture sheets and ``review.csv``. A reviewer gives each row a
verdict:

  real     an object of the taxonomy that the current run no longer masks there;
  junk     not one: the ego body, sky, road, a billboard, a driver inside a
           vehicle, a mask spread over several objects;
  covered  a real object the current run still masks, under another masklet or
           with a mask too different to reach MATCH_IOU (ego pixels trimmed);
  unsure.

Baseline masks at least ``ego_mask.max_fraction`` inside the config's ego mask
are pre-marked junk. Running ``compare`` again keeps the verdicts given and
reports, per class, the baseline masks lost on real objects, with unsure and
unreviewed ones also counted as real for an upper bound.

Into ``out_dir``:

  masklets.csv     one row per masklet of either run: masks sampled, how many
                   the other run matched, and the masklet it matched most;
  review.csv       the baseline masklets with unmatched masks;
  sheets/<n>.png   SHEET_ROWS review rows each: up to SHEET_CROPS unmatched
                   images, the baseline mask in red and the current mask that
                   overlaps it most outlined in green;
  summary.txt      the per-class loss.

It finds what the current run lost, not what both runs miss: an object the
detector never boxes is in neither.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from wato_common.artifacts import masklets
from wato_common.storage import artifact_root
from wato_perception_2d.config import ComponentConfig
from wato_perception_2d.io import load_chunks, load_frame_index

log = logging.getLogger(__name__)

MATCH_IOU = 0.5
VERDICTS = ("real", "junk", "covered", "unsure")
SHEET_ROWS = 8
SHEET_CROPS = 3
CROP_PX = 240
_CAPTION_PX = 18

REVIEW_FIELDS = [
    "row",
    "sheet",
    "chunk_id",
    "cam_id",
    "masklet_id",
    "cls",
    "score",
    "masks",
    "lost_masks",
    "sampled",
    "matched",
    "partner",
    "verdict",
    "note",
]
_EGO_NOTE = "inside the ego mask"


@contextmanager
def _artifact_root(uri: str) -> Iterator[None]:
    """Read from another artifact tree (the baseline run's) for a while."""
    old = os.environ.get("ARTIFACT_ROOT_URI")
    os.environ["ARTIFACT_ROOT_URI"] = uri
    try:
        yield
    finally:
        if old is None:
            del os.environ["ARTIFACT_ROOT_URI"]
        else:
            os.environ["ARTIFACT_ROOT_URI"] = old


@dataclass(frozen=True)
class _Mask:
    """A mask cut down to its bounding box, so IoU only touches overlaps."""

    y0: int
    x0: int
    bits: np.ndarray
    area: int


def _crop(mask: np.ndarray) -> Optional[_Mask]:
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if not len(ys):
        return None
    bits = mask[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
    return _Mask(int(ys[0]), int(xs[0]), bits, int(np.count_nonzero(bits)))


def iou(a: _Mask, b: _Mask) -> float:
    y0, x0 = max(a.y0, b.y0), max(a.x0, b.x0)
    y1 = min(a.y0 + a.bits.shape[0], b.y0 + b.bits.shape[0])
    x1 = min(a.x0 + a.bits.shape[1], b.x0 + b.bits.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    inter = np.count_nonzero(
        a.bits[y0 - a.y0 : y1 - a.y0, x0 - a.x0 : x1 - a.x0]
        & b.bits[y0 - b.y0 : y1 - b.y0, x0 - b.x0 : x1 - b.x0]
    )
    return inter / (a.area + b.area - inter)


def sample(frames: tuple[int, ...], every: int) -> list[int]:
    """A masklet's images on the shared grid (camera_seq % every == 0), so each
    sampled image is read once for all its masklets; a masklet too short to
    reach the grid gets its middle image."""
    on_grid = [s for s in frames if s % every == 0]
    return on_grid or [frames[len(frames) // 2]]


@dataclass
class MaskletMatch:
    """One masklet's sampled masks against the other run's."""

    run: str  # "baseline" | "current"
    chunk_id: str
    record: masklets.MaskletRecord
    sampled: list[int] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    partners: list[Optional[str]] = field(default_factory=list)
    ego_fractions: list[float] = field(default_factory=list)

    @property
    def matched(self) -> int:
        return sum(i >= MATCH_IOU for i in self.ious)

    @property
    def lost_masks(self) -> int:
        """Masks with no match, scaled from the sample to the whole masklet."""
        if not self.sampled:
            return 0
        n = len(self.record.frames_present)
        return round(n * (len(self.sampled) - self.matched) / len(self.sampled))

    @property
    def partner(self) -> str:
        """The other run's masklet matched most often ("" if none)."""
        hits = Counter(p for p, i in zip(self.partners, self.ious) if i >= MATCH_IOU)
        return hits.most_common(1)[0][0] if hits else ""

    @property
    def ego_fraction(self) -> float:
        return float(np.median(self.ego_fractions)) if self.ego_fractions else 0.0


def _match_chunk(
    cfg: Optional[ComponentConfig],
    bag_id: str,
    chunk_id: str,
    roots: dict[str, str],
    every: int,
) -> list[MaskletMatch]:
    from wato_perception_2d.pipeline import _ego_mask

    index: dict[str, masklets.MaskletIndex] = {}
    for run, uri in roots.items():
        with _artifact_root(uri):
            index[run] = masklets.MaskletIndex.load(bag_id, chunk_id)
    matches = {
        (run, r.masklet_id): MaskletMatch(run, chunk_id, r)
        for run, idx in index.items()
        for r in idx.records
    }
    wanted: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for key, m in matches.items():
        for s in sample(m.record.frames_present, every):
            wanted.setdefault((m.record.cam_id, s), set()).add(key)

    ego: dict[str, Optional[np.ndarray]] = {}
    for cam, seq in sorted(wanted):
        masks: dict[str, dict[str, _Mask]] = {}
        for run, idx in index.items():
            masks[run] = {}
            with _artifact_root(roots[run]):
                for r in idx.at(cam, seq):
                    full = idx.mask(r, seq)
                    crop = _crop(full)
                    if crop is None:
                        continue
                    masks[run][r.masklet_id] = crop
                    if run == "baseline" and (run, r.masklet_id) in wanted[(cam, seq)]:
                        if cam not in ego:
                            ego[cam] = (
                                _ego_mask(cfg, cam, full.shape[1], full.shape[0])
                                if cfg is not None
                                else None
                            )
                        region = ego[cam]
                        matches[(run, r.masklet_id)].ego_fractions.append(
                            0.0
                            if region is None
                            else np.count_nonzero(full & region) / crop.area
                        )
        for run, other in (("baseline", "current"), ("current", "baseline")):
            for mid, mine in masks[run].items():
                if (run, mid) not in wanted[(cam, seq)]:
                    continue
                score, partner = max(
                    ((iou(mine, theirs), oid) for oid, theirs in masks[other].items()),
                    default=(0.0, None),
                )
                m = matches[(run, mid)]
                m.sampled.append(seq)
                m.ious.append(score)
                m.partners.append(partner if score > 0 else None)
    return list(matches.values())


def _read_verdicts(path: str) -> dict[str, tuple[str, str]]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        return {
            r["masklet_id"]: (r.get("verdict", "").strip(), r.get("note", ""))
            for r in csv.DictReader(fh)
        }


def review_rows(
    matches: list[MaskletMatch],
    verdicts: dict[str, tuple[str, str]],
    ego_max_fraction: float,
) -> list[dict]:
    """Baseline masklets with unmatched masks, largest loss first. Verdicts
    already given are kept; masklets inside the ego mask are pre-marked junk."""
    lost = sorted(
        (m for m in matches if m.run == "baseline" and m.matched < len(m.sampled)),
        key=lambda m: (-m.lost_masks, m.chunk_id, m.record.cam_id, m.record.masklet_id),
    )
    rows = []
    for i, m in enumerate(lost):
        r = m.record
        verdict, note = verdicts.get(r.masklet_id, ("", ""))
        if not verdict and m.ego_fraction >= ego_max_fraction:
            verdict, note = "junk", _EGO_NOTE
        if verdict and verdict not in VERDICTS:
            raise ValueError(
                f"review.csv: masklet {r.masklet_id} has verdict {verdict!r}; "
                f"use one of {', '.join(VERDICTS)} or leave it empty"
            )
        rows.append(
            {
                "row": i,
                "sheet": f"{i // SHEET_ROWS:03d}.png",
                "chunk_id": m.chunk_id,
                "cam_id": r.cam_id,
                "masklet_id": r.masklet_id,
                "cls": r.cls,
                "score": f"{r.score:.2f}",
                "masks": len(r.frames_present),
                "lost_masks": m.lost_masks,
                "sampled": len(m.sampled),
                "matched": m.matched,
                "partner": m.partner,
                "verdict": verdict,
                "note": note,
            }
        )
    return rows


def summarize(matches: list[MaskletMatch], rows: list[dict]) -> dict[str, dict]:
    """Per class: the baseline's masks, those lost, the lost ones by verdict
    ("unreviewed" when empty), and the current run's masks with no baseline
    match ("added")."""
    out: dict[str, dict] = {}

    def entry(cls: str) -> dict:
        return out.setdefault(
            cls,
            {"masks": 0, "lost": 0, "added": 0, "unreviewed": 0}
            | {v: 0 for v in VERDICTS},
        )

    for m in matches:
        if m.run == "baseline":
            entry(m.record.cls)["masks"] += len(m.record.frames_present)
        else:
            entry(m.record.cls)["added"] += m.lost_masks
    for row in rows:
        e = entry(row["cls"])
        e["lost"] += row["lost_masks"]
        e[row["verdict"] or "unreviewed"] += row["lost_masks"]
    return dict(sorted(out.items()))


def format_summary(summary: dict[str, dict]) -> str:
    """The summary as a table. "real lost" is the real-object share of the
    baseline's masks the current run lost (junk excluded from both); "upper"
    also counts unsure and unreviewed losses as real."""
    head = (
        f"{'class':<12}{'baseline':>9}{'lost':>7}{'real':>6}{'covered':>8}"
        f"{'junk':>6}{'unsure':>7}{'unreviewed':>11}{'real lost':>10}"
        f"{'upper':>7}{'added':>7}"
    )
    lines = [head]
    for cls, e in summary.items():
        base = e["masks"] - e["junk"]

        def pct(n: int) -> str:
            return f"{100 * n / base:.1f}%" if base > 0 else "-"

        lines.append(
            f"{cls:<12}{e['masks']:>9}{e['lost']:>7}{e['real']:>6}{e['covered']:>8}"
            f"{e['junk']:>6}{e['unsure']:>7}{e['unreviewed']:>11}"
            f"{pct(e['real']):>10}{pct(e['real'] + e['unsure'] + e['unreviewed']):>7}"
            f"{e['added']:>7}"
        )
    return "\n".join(lines) + "\n"


def _outline(mask: np.ndarray) -> np.ndarray:
    """A 2-pixel border of ``mask``."""
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    edge = mask & ~inner
    edge[1:] |= edge[:-1].copy()
    edge[:, 1:] |= edge[:, :-1].copy()
    return edge


def tile(
    image: np.ndarray,
    baseline: np.ndarray,
    current: Optional[np.ndarray],
    caption: str,
):
    """A CROP_PX square: the image around the baseline mask (red), with the
    current mask outlined in green; small masks are enlarged."""
    from PIL import Image, ImageDraw

    ys, xs = np.nonzero(baseline)
    h, w = baseline.shape
    pad = max(24, (max(np.ptp(ys), np.ptp(xs)) + 1) // 2)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    scale = CROP_PX / max(y1 - y0, x1 - x0)
    size = (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale)))

    def fit(a: np.ndarray, resample) -> np.ndarray:
        return np.asarray(Image.fromarray(a[y0:y1, x0:x1]).resize(size, resample))

    rgb = fit(image, Image.BILINEAR).copy()
    red = fit(baseline.astype(np.uint8) * 255, Image.NEAREST) > 0
    rgb[red] = (0.55 * rgb[red] + [115, 0, 0]).astype(np.uint8)
    if current is not None:
        green = _outline(fit(current.astype(np.uint8) * 255, Image.NEAREST) > 0)
        rgb[green] = (0, 255, 0)
    out = Image.new("RGB", (CROP_PX, CROP_PX))
    out.paste(
        Image.fromarray(rgb), ((CROP_PX - size[0]) // 2, (CROP_PX - size[1]) // 2)
    )
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, CROP_PX - 14, CROP_PX, CROP_PX), fill=(0, 0, 0))
    draw.text((3, CROP_PX - 13), caption, fill=(255, 255, 0))
    return out


def _write_sheets(
    bag_id: str,
    rows: list[dict],
    matches: dict[str, MaskletMatch],
    roots: dict[str, str],
    out_dir: str,
) -> None:
    from PIL import Image, ImageDraw

    sheet_dir = os.path.join(out_dir, "sheets")
    os.makedirs(sheet_dir, exist_ok=True)
    images: dict[str, dict[tuple[str, int], str]] = {}
    row_px = _CAPTION_PX + CROP_PX
    for start in range(0, len(rows), SHEET_ROWS):
        batch = rows[start : start + SHEET_ROWS]
        sheet = Image.new("RGB", (SHEET_CROPS * CROP_PX, len(batch) * row_px))
        draw = ImageDraw.Draw(sheet)
        for k, row in enumerate(batch):
            m = matches[row["masklet_id"]]
            chunk = m.chunk_id
            if chunk not in images:
                images[chunk] = {
                    (f.cam_id, f.camera_seq): f.image_path
                    for f in load_frame_index(bag_id, chunk)
                }
            draw.text(
                (3, k * row_px + 3),
                f"#{row['row']} {chunk}/{row['cam_id']} {row['masklet_id'][:8]} "
                f"{row['cls']} {row['score']}  lost {row['lost_masks']} of "
                f"{row['masks']}  {row['verdict']}",
                fill=(255, 255, 255),
            )
            unmatched = [
                (s, p, i)
                for s, p, i in zip(m.sampled, m.partners, m.ious)
                if i < MATCH_IOU
            ]
            picks = sorted({0, len(unmatched) // 2, len(unmatched) - 1})
            for j, (seq, partner, score) in enumerate(
                unmatched[p] for p in picks[:SHEET_CROPS]
            ):
                with _artifact_root(roots["baseline"]):
                    base = masklets.read_mask(bag_id, chunk, row["masklet_id"], seq)
                cur = None
                if partner is not None:
                    with _artifact_root(roots["current"]):
                        cur = masklets.read_mask(bag_id, chunk, partner, seq)
                img = np.asarray(
                    Image.open(images[chunk][(row["cam_id"], seq)]).convert("RGB")
                )
                sheet.paste(
                    tile(img, base, cur, f"seq {seq}  IoU {score:.2f}"),
                    (j * CROP_PX, k * row_px + _CAPTION_PX),
                )
        sheet.save(os.path.join(sheet_dir, f"{start // SHEET_ROWS:03d}.png"))


def run(
    cfg: Optional[ComponentConfig],
    *,
    bag_id: str,
    baseline_uri: str,
    out_dir: str,
    chunk_ids: Optional[list[str]] = None,
    every: int = 5,
) -> dict[str, dict]:
    """Compare the current artifact tree's masklets for ``bag_id`` with those
    under ``baseline_uri``; write the files above to ``out_dir`` and return the
    summary. ``cfg`` is the current run's config (its ego mask pre-marks junk).
    """
    roots = {"baseline": baseline_uri, "current": artifact_root()}
    if not chunk_ids:
        chunk_ids = [str(c["chunk_id"]) for c in load_chunks(bag_id)]
    both = []
    for chunk_id in chunk_ids:
        present = []
        for uri in roots.values():
            with _artifact_root(uri):
                present.append(masklets.exists(bag_id, chunk_id))
        if all(present):
            both.append(chunk_id)
    if not both:
        raise FileNotFoundError(
            f"no chunk of {bag_id} has masklets in both {baseline_uri} and "
            f"{roots['current']}"
        )

    matches: list[MaskletMatch] = []
    for chunk_id in both:
        found = _match_chunk(cfg, bag_id, chunk_id, roots, every)
        log.info(
            "chunk %s: %d baseline / %d current masklets",
            chunk_id,
            sum(m.run == "baseline" for m in found),
            sum(m.run == "current" for m in found),
        )
        matches += found

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "masklets.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["run", "chunk_id", "cam_id", "masklet_id", "cls", "score", "masks"]
            + ["sampled", "matched", "partner", "ego_fraction"]
        )
        for m in matches:
            r = m.record
            w.writerow(
                [m.run, m.chunk_id, r.cam_id, r.masklet_id, r.cls, f"{r.score:.3f}"]
                + [len(r.frames_present), len(m.sampled), m.matched, m.partner]
                + [f"{m.ego_fraction:.3f}"]
            )

    review_path = os.path.join(out_dir, "review.csv")
    max_fraction = cfg.ego_mask.max_fraction if cfg is not None else 1.0
    rows = review_rows(matches, _read_verdicts(review_path), max_fraction)
    with open(review_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    _write_sheets(
        bag_id,
        rows,
        {m.record.masklet_id: m for m in matches if m.run == "baseline"},
        roots,
        out_dir,
    )

    summary = summarize(matches, rows)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(
            f"# {bag_id}, chunks {', '.join(both)}: current run against "
            f"{baseline_uri}, every {every}th image; masks lost at IoU < "
            f"{MATCH_IOU}\n"
        )
        fh.write(format_summary(summary))
    return summary
