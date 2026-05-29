"""Phrase deduplication for Florence-2 + SAM3 output.

Florence-2 emits synonyms for the same object: "car", "vehicle", "sedan".
SAM3 faithfully returns overlapping masks for each.  Two-pass cleanup:

1. coarsen_synonyms(): cluster phrases by CLIP text-embedding cosine
   similarity and keep the highest-scored mask per cluster.
2. nms_3d(): back-project mask centroids to world using metric depth,
   then cluster by 3D proximity (1.0 m radius).  Falls back to 2D IoU NMS
   when depth alignment failed (depth map is all zeros).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from wato_perception_2d.models.segmenter import SegmentedDetection

log = logging.getLogger(__name__)


def coarsen_synonyms(
    seg_detections: list[SegmentedDetection],
    clip_threshold: float = 0.85,
) -> list[SegmentedDetection]:
    """Merge detections whose phrases are semantically equivalent.

    Groups phrases by CLIP text-embedding cosine similarity.  Within each
    group keeps the detection with the highest sam3_score.

    Falls back to exact-string deduplication if CLIP (via transformers) is
    unavailable.

    Args:
        seg_detections: list of SegmentedDetection (one per SAM3 output).
        clip_threshold: cosine similarity threshold for merging [0, 1].

    Returns:
        Deduplicated list (at most one detection per semantic cluster).
    """
    if len(seg_detections) <= 1:
        return seg_detections

    embeddings = _clip_text_embeddings([d.phrase for d in seg_detections])
    if embeddings is None:
        return _exact_string_dedup(seg_detections)

    n = len(seg_detections)
    assigned = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= clip_threshold:
                assigned[j] = cluster_id
        cluster_id += 1

    # Within each cluster, keep the detection with highest sam3_score.
    best: dict[int, SegmentedDetection] = {}
    for det, cid in zip(seg_detections, assigned):
        if cid not in best or det.sam3_score > best[cid].sam3_score:
            best[cid] = det

    return list(best.values())


def _exact_string_dedup(
    seg_detections: list[SegmentedDetection],
) -> list[SegmentedDetection]:
    """Deduplicate by exact phrase string, keeping highest sam3_score."""
    best: dict[str, SegmentedDetection] = {}
    for det in seg_detections:
        if det.phrase not in best or det.sam3_score > best[det.phrase].sam3_score:
            best[det.phrase] = det
    return list(best.values())


def _clip_text_embeddings(
    phrases: list[str],
) -> Optional[np.ndarray]:
    """Return (N, D) normalized CLIP text embeddings, or None if unavailable."""
    try:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

        inputs = tokenizer(phrases, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embeds = outputs.last_hidden_state[:, 0, :]  # CLS token
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds.cpu().numpy()
    except Exception:  # noqa: BLE001
        return None


def nms_3d(
    seg_detections: list[SegmentedDetection],
    depth: np.ndarray,
    K: np.ndarray,
    world_T_ego: np.ndarray,
    ego_T_cam: np.ndarray,
    radius_m: float = 1.0,
) -> list[SegmentedDetection]:
    """Remove duplicate detections by 3D centroid proximity.

    Back-projects each mask centroid to world frame using the provided metric
    depth map.  Clusters by 3D Euclidean distance; within each cluster keeps
    the detection with the highest sam3_score.

    Falls back to nms_2d_fallback() when the depth map is all-zero (failed fit).

    Args:
        seg_detections: list of SegmentedDetection after coarsen_synonyms().
        depth: (H, W) float16/float32 metric depth in metres.
        K: (3, 3) intrinsic matrix.
        world_T_ego: (4, 4) SE3 ego pose in world frame.
        ego_T_cam: (4, 4) SE3 camera extrinsic (cam → ego frame).
        radius_m: cluster radius in world-frame metres.

    Returns:
        Deduplicated list.
    """
    if len(seg_detections) <= 1:
        return seg_detections

    depth_arr = np.asarray(depth, dtype=np.float32)
    if float(depth_arr.max()) == 0.0:
        return nms_2d_fallback(seg_detections)

    from wato_common.geometry import lift_pixel_to_world

    H, W = depth_arr.shape

    # Back-project mask centroid for each detection.
    world_pts: list[Optional[np.ndarray]] = []
    for det in seg_detections:
        cx, cy = _mask_centroid(det.mask)
        if cx is None:
            world_pts.append(None)
            continue
        u, v = int(round(cx)), int(round(cy))
        u, v = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
        d = float(depth_arr[v, u])
        if d <= 0:
            world_pts.append(None)
            continue
        pt = lift_pixel_to_world(float(u), float(v), d, K, world_T_ego, ego_T_cam)
        world_pts.append(pt)

    n = len(seg_detections)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        if world_pts[i] is None:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if world_pts[j] is None:
                continue
            dist = float(np.linalg.norm(world_pts[i] - world_pts[j]))
            if dist < radius_m:
                # Keep the higher-scored detection.
                if seg_detections[i].sam3_score >= seg_detections[j].sam3_score:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [det for det, k in zip(seg_detections, keep) if k]


def nms_2d_fallback(
    seg_detections: list[SegmentedDetection],
    iou_threshold: float = 0.5,
) -> list[SegmentedDetection]:
    """Standard 2D IoU NMS over mask bounding boxes.

    Used when metric depth is unavailable.  Sorts by sam3_score descending,
    greedily suppresses lower-scored overlapping detections.
    """
    if len(seg_detections) <= 1:
        return seg_detections

    sorted_dets = sorted(seg_detections, key=lambda d: d.sam3_score, reverse=True)
    keep_mask = [True] * len(sorted_dets)

    for i in range(len(sorted_dets)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(sorted_dets)):
            if not keep_mask[j]:
                continue
            if _mask_iou(sorted_dets[i].mask, sorted_dets[j].mask) >= iou_threshold:
                keep_mask[j] = False

    return [det for det, k in zip(sorted_dets, keep_mask) if k]


def _mask_centroid(mask: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Return (cx, cy) pixel centroid of a bool mask, or (None, None) if empty."""
    ys, xs = np.where(mask)
    if xs.shape[0] == 0:
        return None, None
    return float(xs.mean()), float(ys.mean())


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0
