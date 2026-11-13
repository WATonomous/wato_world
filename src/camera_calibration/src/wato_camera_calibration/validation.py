"""Independent static correspondences, scene-block diagnostics and decisions.

Image/intensity edge distance is diagnostic until qualified against independent
correspondences. A low NID objective or an arbitrary pixel limit cannot certify
a physical mount. Validation/test observations never select the optimized fit.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    map_coordinates,
    sobel,
)
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from wato_camera_calibration.config import SolverConfig, ValidationConfig
from wato_camera_calibration.inputs import Sample, world_to_ego


def change(reference, candidate) -> dict:
    return {
        "translation_m": float(np.linalg.norm(candidate[:3, 3] - reference[:3, 3])),
        "rotation_deg": float(
            np.rad2deg(
                Rotation.from_matrix(
                    reference[:3, :3].T @ candidate[:3, :3]
                ).magnitude()
            )
        ),
    }


def stability(transforms: list[np.ndarray]) -> dict:
    changes = [
        change(a, b) for i, a in enumerate(transforms) for b in transforms[i + 1 :]
    ]
    return {
        "rotation_spread_deg": max((c["rotation_deg"] for c in changes), default=None),
        "translation_spread_m": max(
            (c["translation_m"] for c in changes), default=None
        ),
        "fits": len(transforms),
    }


def edge_metrics(samples: list[Sample], cam, transform) -> dict:
    values, coverage, depths = [], set(), []
    for sample in samples:
        smooth = gaussian_filter(sample.gray.astype(float), 1)
        gradient = np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1))
        threshold = max(10.0, float(np.percentile(gradient, 90)))
        edges = gradient >= threshold
        if not edges.any():
            continue
        distance = distance_transform_edt(~edges)
        # Select intensity boundaries in 3D independently of candidate projection.
        _, neighbors = cKDTree(sample.xyz).query(sample.xyz, k=min(6, len(sample.xyz)))
        contrast = np.ptp(sample.intensity[neighbors], axis=1)
        features = contrast >= max(0.1, float(np.percentile(contrast, 75)))
        uv, z, ok = cam.model.project(
            sample.xyz,
            np.linalg.inv(transform),
            (sample.gray.shape[1], sample.gray.shape[0]),
        )
        finite = np.isfinite(uv).all(axis=1)
        safe = np.where(finite[:, None], uv, 0)
        residual = map_coordinates(
            distance, [safe[:, 1], safe[:, 0]], order=1, mode="constant", cval=50.0
        )
        residual[~ok] = 50.0
        if features.any():
            values.append(float(np.mean(np.minimum(residual[features], 50.0))))
        h, w = sample.gray.shape
        cells = np.floor(safe[ok] / np.array([w / 4, h / 4])).astype(int)
        coverage.update(tuple(c) for c in cells)
        depths.extend(z[ok].tolist())
    return {
        "views": len(values),
        "mean_edge_distance_px": float(np.mean(values)) if values else None,
        "coverage_fraction": len(coverage) / 16,
        "depth_ratio": float(
            np.percentile(depths, 90) / max(np.percentile(depths, 10), 0.5)
        )
        if depths
        else None,
    }


def correspondence_residuals(samples, cam, transform, annotations: dict) -> dict:
    by_time = {s.timestamp_ns: s for s in samples}
    residuals = defaultdict(list)
    for row in annotations.get(cam.cam_id, []):
        sample = by_time.get(int(row["timestamp_ns"]))
        if sample is None or sample.split == "train":
            continue
        if row.get("provenance") != "independent_static":
            raise ValueError(
                "correspondences must declare independent_static provenance"
            )
        xyz = np.asarray(row["world_points"], dtype=float)
        pixels = np.asarray(row["pixels"], dtype=float)
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or pixels.shape != (len(xyz), 2)
            or not len(xyz)
        ):
            raise ValueError(
                "correspondence points/pixels must have matching Nx3/Nx2 shapes"
            )
        if not np.isfinite(xyz).all() or not np.isfinite(pixels).all():
            raise ValueError("correspondences contain nonfinite values")
        uv, _, ok = cam.model.project(
            world_to_ego(xyz, sample.world_T_ego),
            np.linalg.inv(transform),
            (sample.gray.shape[1], sample.gray.shape[0]),
        )
        errors = np.linalg.norm(uv - pixels, axis=1)
        errors[~ok] = max(sample.gray.shape)
        residuals[sample.split].append((sample.block, errors))
    return dict(residuals)


def residual_summary(rows: list) -> dict:
    values = np.concatenate([r[1] for r in rows]) if rows else np.array([])
    return {
        "n": len(values),
        "blocks": len({r[0] for r in rows}),
        "median_px": float(np.median(values)) if len(values) else None,
        "p90_px": float(np.percentile(values, 90)) if len(values) else None,
    }


def improvement_interval(baseline, candidate):
    gains = defaultdict(list)
    for (block, a), (other, b) in zip(baseline, candidate, strict=True):
        if block != other or len(a) != len(b):
            raise ValueError("validation correspondences changed during comparison")
        gains[block].extend((a - b).tolist())
    means = np.array([np.mean(v) for v in gains.values()])
    if len(means) < 2:
        return None
    boots = (
        np.random.default_rng(0)
        .choice(means, (2000, len(means)), replace=True)
        .mean(axis=1)
    )
    return np.percentile(boots, [2.5, 97.5]).tolist()


def evaluate(
    samples,
    cam,
    candidate,
    transforms,
    cfg: ValidationConfig,
    solver: SolverConfig,
    annotations: dict,
) -> dict:
    counts = Counter(s.split for s in samples)
    blocks = {
        split: len({s.block for s in samples if s.split == split})
        for split in ("train", "validation", "test")
    }
    original = cam.ego_T_cam
    base_errors = correspondence_residuals(samples, cam, original, annotations)
    new_errors = correspondence_residuals(samples, cam, candidate, annotations)
    baseline, refined = {}, {}
    improvement = {}
    for split in ("validation", "test"):
        subset = [s for s in samples if s.split == split]
        baseline[split] = {
            "correspondences": residual_summary(base_errors.get(split, [])),
            "edges": edge_metrics(subset, cam, original),
        }
        refined[split] = {
            "correspondences": residual_summary(new_errors.get(split, [])),
            "edges": edge_metrics(subset, cam, candidate),
        }
        improvement[split] = improvement_interval(
            base_errors.get(split, []), new_errors.get(split, [])
        )
    spread = stability(transforms)
    delta = change(original, candidate)
    enough = all(
        counts[s] >= cfg.min_views_per_split and blocks[s] >= cfg.min_blocks_per_split
        for s in blocks
    )
    evidence = all(
        baseline[s]["correspondences"]["n"] >= cfg.min_correspondences
        and baseline[s]["correspondences"]["blocks"] >= cfg.min_blocks_per_split
        for s in baseline
    )

    def absolute(metrics):
        return all(
            m["correspondences"]["median_px"] is not None
            and m["correspondences"]["median_px"] <= cfg.max_reprojection_median_px
            and m["correspondences"]["p90_px"] <= cfg.max_reprojection_p90_px
            and m["edges"]["coverage_fraction"] >= cfg.min_coverage_fraction
            and (m["edges"]["depth_ratio"] or 0) >= cfg.min_depth_ratio
            for m in metrics.values()
        )

    gates = {
        "qualified_profile": cfg.qualified,
        "independent_views": enough,
        "independent_correspondences": evidence,
        "absolute_alignment": evidence and absolute(refined),
        "stable": spread["fits"] >= 2
        and spread["rotation_spread_deg"] is not None
        and spread["rotation_spread_deg"] <= cfg.max_rotation_spread_deg
        and spread["translation_spread_m"] <= cfg.max_translation_spread_m,
        "plausible": delta["rotation_deg"] < 0.95 * solver.max_rotation_deg
        and delta["translation_m"] < 0.95 * solver.max_translation_m,
        "held_out_improvement": evidence
        and all(
            baseline[s]["correspondences"]["median_px"]
            - refined[s]["correspondences"]["median_px"]
            >= cfg.min_improvement_px
            and improvement[s] is not None
            and improvement[s][0] > 0
            for s in baseline
        ),
        "no_edge_regression": all(
            baseline[s]["edges"]["mean_edge_distance_px"] is not None
            and refined[s]["edges"]["mean_edge_distance_px"] is not None
            and refined[s]["edges"]["mean_edge_distance_px"]
            <= baseline[s]["edges"]["mean_edge_distance_px"]
            + cfg.max_edge_regression_px
            for s in baseline
        ),
    }
    if all(gates.values()):
        status = "refined"
    elif cfg.qualified and enough and evidence and absolute(baseline):
        status = "recorded_valid"
    elif not cfg.qualified or not enough or not evidence or not gates["stable"]:
        status = "insufficient_evidence"
    else:
        status = "rejected"
    return {
        "status": status,
        "estimated_parameters": ["rotation"]
        if solver.motion_model == "rotation_only"
        else ["rotation", "translation"],
        "gates": gates,
        "views": dict(counts),
        "blocks": blocks,
        "change": delta,
        "stability": spread,
        "baseline": baseline,
        "candidate": refined,
        "improvement_bootstrap_95_px": improvement,
        "reasons": [name for name, passed in gates.items() if not passed],
    }
