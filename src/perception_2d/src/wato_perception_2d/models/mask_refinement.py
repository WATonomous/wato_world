"""Image-mask seeds using the pinned SAM2 APIs, sharing the video model.

The low-resolution-logit recurrence follows SAM2's image predictor / automatic
mask generator; the distance-interior click is inspired by SAM-PD and
SAMRefiner. This is a small experimental policy, not a reproduction of either
paper. Scores measure model confidence / threshold stability, not correctness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from wato_perception_2d.config import MaskRefinementConfig


def stability_scores(logits, offset: float = 1.0) -> np.ndarray:
    """SAM2 AMG stability: area(logits > offset) / area(logits > -offset).

    Reduce on the tensor's device before moving the small counts to CPU.
    Empty masks score zero; integer reductions avoid overflow on large images.
    """
    n = logits.shape[0]
    if n == 0:
        return np.empty(0, dtype=float)
    high = (logits > offset).reshape(n, -1).sum(axis=1)
    low = (logits > -offset).reshape(n, -1).sum(axis=1)
    if hasattr(high, "cpu"):
        high, low = high.cpu().numpy(), low.cpu().numpy()
    return np.divide(high, low, out=np.zeros(n, dtype=float), where=low > 0)


def interior_point(mask: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """An (x,y) foreground pixel farthest from the boundary, inside the box.

    Pad the image with background so objects touching image edges have a real
    distance to that boundary. Restrict the result to the target's box, rather
    than clicking a possibly larger leaked region elsewhere in the image.
    """
    h, w = mask.shape
    x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), (w, h))
    if x2 <= x1 or y2 <= y1:
        return None
    distance = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    region = distance[y1:y2, x1:x2]
    if not region.any():
        return None
    y, x = np.unravel_index(np.argmax(region), region.shape)
    return np.array([[x + x1, y + y1]], dtype=np.float32)


@dataclass
class RefinedMask:
    mask: np.ndarray | None
    predicted_iou: float | None
    stability: float | None
    reason: str
    used_interior_point: bool = False
    used_mask_input: bool = False
    initial_candidate_index: int | None = None

    def diagnostics(self) -> dict:
        return {
            "predicted_iou": self.predicted_iou,
            "stability": self.stability,
            "reason": self.reason,
            "used_interior_point": self.used_interior_point,
            "used_mask_input": self.used_mask_input,
            "initial_candidate_index": self.initial_candidate_index,
        }


class KeyframeMaskRefiner:
    """Encode a keyframe once, propose masks, then release image features.

    Public image-predictor methods only. Constructing the image predictor
    references the video model's existing weights; it does not load a second
    checkpoint. Call set_image only when that keyframe actually needs a seed
    or an eligible correction. Call reset after its prompts / on any failure.
    """

    def __init__(self, video_predictor, config: MaskRefinementConfig) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.predictor = SAM2ImagePredictor(
            getattr(video_predictor, "image_model", video_predictor)
        )
        self.config = config

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.predictor.set_image(image_rgb)

    def reset(self) -> None:
        self.predictor.reset_predictor()

    def box_mask(self, box_xyxy: tuple[float, float, float, float]) -> np.ndarray:
        """SAM2's single-output mask (logits > 0) for a box on the encoded
        image: a detection's mask, as SAM 3 matches detections to masklets by
        mask IoU (its App. F.5 prompts SAM with an external detector's boxes)."""
        logits, _, _ = self.predictor.predict(
            box=np.asarray(box_xyxy, dtype=np.float32),
            multimask_output=False,
            return_logits=True,
        )
        return logits[0] > 0

    def propose(
        self, box_xyxy: tuple[float, float, float, float], ego_mask=None
    ) -> RefinedMask:
        box = np.asarray(box_xyxy, dtype=np.float32)
        logits, scores, low_res = self.predictor.predict(
            box=box,
            multimask_output=self.config.multimask_output,
            return_logits=True,
        )
        if low_res.ndim != 3 or low_res.shape[0] != len(scores):
            raise ValueError(
                "SAM2 image predictor returned invalid low-resolution logits"
            )
        selected = self._select(logits, scores, ego_mask, box)
        if selected is None:
            return RefinedMask(None, None, None, "no_acceptable_candidate")
        idx, result = selected
        if result.mask is None or not (
            self.config.interior_point or self.config.mask_input_refinement
        ):
            return result
        # This point derives from the coarse mask, not independent supervision.
        # Retain the coarse candidate if the extra decoder pass lowers quality.
        point = interior_point(result.mask, box) if self.config.interior_point else None
        logits2, scores2, _ = self.predictor.predict(
            box=box,
            point_coords=point,
            point_labels=np.ones(1, dtype=np.int32) if point is not None else None,
            mask_input=low_res[idx : idx + 1]
            if self.config.mask_input_refinement
            else None,
            multimask_output=False,
            return_logits=True,
        )
        refined = self._select(logits2, scores2, ego_mask, box)
        if refined is not None:
            _, candidate = refined
            if (
                candidate.mask is not None
                and candidate.predicted_iou > result.predicted_iou
                and candidate.stability >= result.stability
                and (
                    point is None or candidate.mask[int(point[0, 1]), int(point[0, 0])]
                )
            ):
                candidate.used_interior_point = point is not None
                candidate.used_mask_input = self.config.mask_input_refinement
                candidate.initial_candidate_index = idx
                return candidate
        return result

    def _select(self, logits, scores, ego_mask, box) -> tuple[int, RefinedMask] | None:
        if logits.ndim != 3 or np.asarray(scores).shape != (logits.shape[0],):
            raise ValueError(
                "SAM2 image predictor returned invalid mask / score shapes"
            )
        stability = stability_scores(logits, self.config.stability_offset)
        rejected = None
        for idx in np.argsort(-np.asarray(scores), kind="stable"):
            score, stable = float(scores[idx]), float(stability[idx])
            mask = logits[idx] > 0
            h, w = mask.shape
            x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
            x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), (w, h))
            reason = "accepted"
            if not np.isfinite(score) or not np.isfinite(logits[idx]).all():
                reason = "nonfinite_candidate"
            elif not mask.any():
                reason = "empty_candidate"
            elif score < self.config.min_predicted_iou:
                reason = "low_predicted_iou"
            elif stable < self.config.min_stability:
                reason = "low_stability"
            elif x2 <= x1 or y2 <= y1 or not mask[y1:y2, x1:x2].any():
                reason = "outside_prompt_box"
            # Do not turn a mask of the ego vehicle into a trusted seed by
            # trimming most of it away. Let the baseline box path handle it.
            if ego_mask is not None and np.any(mask & ego_mask):
                reason = "ego_overlap"
            result = RefinedMask(
                mask if reason == "accepted" else None,
                score if np.isfinite(score) else None,
                stable,
                reason,
                initial_candidate_index=int(idx),
            )
            if result.mask is not None:
                return int(idx), result
            if rejected is None:
                rejected = int(idx), result
        return rejected
