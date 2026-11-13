"""Baseline/candidate overlays; colours represent optical depth."""

from pathlib import Path
from html import escape

import numpy as np
from PIL import Image, ImageDraw


def overlay(samples, cam, candidate, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        if sample.split == "train":
            continue
        views = []
        for label, transform in (("Recorded", cam.ego_T_cam), ("Candidate", candidate)):
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            uv, z, ok = cam.model.project(
                sample.xyz, np.linalg.inv(transform), image.size
            )
            indices = np.flatnonzero(ok)[::2]
            for i in indices:
                depth = min(float(z[i]) / 50, 1)
                colour = (int(255 * (1 - depth)), int(180 * depth), int(255 * depth))
                x, y = uv[i]
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=colour)
            draw.rectangle((0, 0, image.width, 28), fill="black")
            draw.text(
                (8, 7),
                f"{label} | {sample.split} | {sample.timestamp_ns}",
                fill="white",
            )
            views.append(image)
        combined = Image.new("RGB", (views[0].width * 2, views[0].height))
        for i, view in enumerate(views):
            combined.paste(view, (i * view.width, 0))
        combined.thumbnail((1600, 800))
        combined.save(root / f"{sample.timestamp_ns}.jpg", quality=90)


def write_index(root: Path, report: dict) -> Path:
    """A local diagnostic gallery; all geometry remains in the report artifact."""
    cards = []
    for cid, row in sorted(report["cameras"].items()):
        images = (
            sorted((root / "diagnostics" / cid).glob("*.jpg"))
            if row["status"] not in ("not_requested", "error")
            else []
        )
        previews = "".join(
            f'<a href="{escape(str(p.relative_to(root)), quote=True)}"><img loading="lazy" '
            f'src="{escape(str(p.relative_to(root)), quote=True)}" alt="{escape(cid)} {p.stem}"></a>'
            for p in images
        )
        cards.append(
            f"<section><h2>{escape(cid)}: {escape(row['status'])}</h2>"
            f"<p>{escape(', '.join(row.get('reasons', [])))}</p>{previews}</section>"
        )
    target = root / "index.html"
    target.write_text(
        '<!doctype html><meta charset="utf-8"><title>Camera calibration diagnostics</title>'
        "<style>body{font:16px system-ui;max-width:1600px;margin:2em auto;padding:1em;background:#111;color:#eee}"
        "img{max-width:100%;margin:.5em 0}section{margin-bottom:3em}a{color:#9cf}</style>"
        f"<h1>Camera calibration: {escape(report['bag_id'])}</h1>"
        "<p>Recorded / candidate overlays. Candidate alignment is not proof of calibration. "
        '<a href="report.json">Full report</a></p>' + "".join(cards),
        encoding="utf-8",
    )
    return target
