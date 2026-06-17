"""Florence-2 open-vocabulary region discovery.

Runs the <DENSE_REGION_CAPTION> task on each image to produce noun phrases
with rough bounding boxes.  These phrases are then passed to SAM3 for
text-prompted segmentation, replacing the GroundingDINO fixed-vocabulary
detection step.

Lazy-imports Florence-2 (via transformers) so the module can be *imported*
without it installed; calling propose() without it raises loudly.  There is no
degraded fallback — use discovery.backend: "fixed" (FixedClassDiscovery) to run
without Florence-2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RegionProposal:
    """One region proposal from Florence-2 dense-region captioning."""

    phrase: str
    rough_box: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    confidence: float


class Florence2Discovery:
    """Florence-2 wrapper for open-vocabulary noun-phrase discovery.

    Uses the <DENSE_REGION_CAPTION> task to extract noun phrases and rough
    bounding boxes from an image without a fixed class vocabulary.

    Raises RuntimeError if Florence-2 (transformers) is unavailable — no
    degraded fallback.
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large-ft",
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._device = device or self._default_device()
        self._model = None
        self._processor = None

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self._model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id, trust_remote_code=True
            ).to(self._device)
            self._model.eval()
            log.info("Florence-2 loaded (%s) on %s", self._model_id, self._device)
        except Exception as exc:
            raise RuntimeError(
                f"Florence-2 model {self._model_id!r} could not be loaded: {exc}. "
                "Install 'transformers' or set discovery.backend: 'fixed' to run "
                "with a closed-set class list. perception_2d does not fall back."
            ) from exc

    def propose(self, image_rgb: np.ndarray) -> list[RegionProposal]:
        """Run <DENSE_REGION_CAPTION> on one image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.

        Returns:
            List of RegionProposal, each with a noun phrase, rough bounding
            box (x1, y1, x2, y2) in pixels, and a confidence score.

        Raises:
            RuntimeError: if Florence-2 is not installed/loadable.
        """
        self._load()

        import torch
        from PIL import Image as PILImage

        task = "<DENSE_REGION_CAPTION>"
        pil_img = PILImage.fromarray(image_rgb)
        inputs = self._processor(text=task, images=pil_img, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        result = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image_rgb.shape[1], image_rgb.shape[0]),
        )
        return _parse_florence_output(result, task)


class FixedClassDiscovery:
    """Closed-vocabulary 'discovery' that bypasses Florence-2.

    Emits one RegionProposal per configured class name, each spanning the whole
    image, so SAM 3.1 is text-prompted directly with the class names (e.g. COCO
    classes) instead of with Florence-2's open-vocabulary phrases.  Exposes the
    same ``propose(image)`` interface as Florence2Discovery so the pipeline can
    swap backends without other changes.
    """

    def __init__(self, classes: list[str]) -> None:
        self._classes = [c.strip() for c in classes if c and c.strip()]
        if not self._classes:
            log.warning(
                "FixedClassDiscovery initialised with no classes — "
                "perception_2d will produce no detections."
            )

    def propose(self, image_rgb: np.ndarray) -> list[RegionProposal]:
        """Return one full-image proposal per fixed class."""
        H, W = image_rgb.shape[:2]
        box = (0.0, 0.0, float(W), float(H))
        return [
            RegionProposal(phrase=cls, rough_box=box, confidence=1.0)
            for cls in self._classes
        ]


def _parse_florence_output(result: dict, task: str) -> list[RegionProposal]:
    """Convert Florence-2 post-processed output into RegionProposal list."""
    proposals: list[RegionProposal] = []
    task_result = result.get(task, {})
    labels = task_result.get("labels", [])
    bboxes = task_result.get("bboxes", [])

    for phrase, box in zip(labels, bboxes):
        phrase = phrase.strip()
        if not phrase:
            continue
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        proposals.append(
            RegionProposal(phrase=phrase, rough_box=(x1, y1, x2, y2), confidence=1.0)
        )
    return proposals
