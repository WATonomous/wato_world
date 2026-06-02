"""Lazy torch + MF-MOS imports for inference.

The ONLY module in this package that imports torch or the MF-MOS vendored
code. Imported lazily by mf_mos._core._load_model() so test collection and
CPU-only pipeline runs don't pay torch startup cost.

Adds third_party/MF-MOS to sys.path on first import (idempotent).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)

_MFMOS_ROOT = "/opt/mf_mos"

if _MFMOS_ROOT not in sys.path:
    sys.path.insert(0, _MFMOS_ROOT)


class MFMosModel:
    """Thin wrapper around the upstream MFMOS nn.Module.

    Builds the model at its training resolution (e.g. 64×2048 for KITTI),
    loads the checkpoint, normalises inputs using img_means/img_stds from
    the data config, resizes between sensor and model resolution, and
    pads/truncates residual channels to match n_input_scans.
    """

    def __init__(
        self,
        checkpoint_path: str,
        arch_cfg: str,
        data_cfg: str,
        device: str = "cuda",
    ) -> None:
        import torch

        with open(arch_cfg, "r", encoding="utf-8") as fh:
            self.ARCH: dict[str, Any] = yaml.safe_load(fh)
        with open(data_cfg, "r", encoding="utf-8") as fh:
            self.DATA: dict[str, Any] = yaml.safe_load(fh)

        sensor = self.ARCH["dataset"]["sensor"]
        self.H_model: int = int(sensor["img_prop"]["height"])
        self.W_model: int = int(sensor["img_prop"]["width"])
        self.n_input_scans: int = int(sensor["n_input_scans"])
        self.img_means: list[float] = sensor["img_means"]  # 5 values
        self.img_stds: list[float] = sensor["img_stds"]  # 5 values
        self.res_mean: float = float(sensor.get("res_mean", 0.0))
        self.res_std: float = float(sensor.get("res_std", 1.0))

        self.device = torch.device(device)

        # Inference uses batch_size=1 regardless of training batch_size.
        from modules.MFMOS import MFMOS  # type: ignore[import]

        moving_map_inv = self.DATA.get("moving_learning_map_inv", {0: 0, 1: 251})
        movable_map_inv = self.DATA.get("movable_learning_map_inv", {0: 0, 1: 250})
        nclasses = len(moving_map_inv)
        movable_nclasses = len(movable_map_inv)
        # The "moving" class index = the key whose semantic-label value is
        # >=251. Falls back to the highest key for 2-class models.
        self._moving_class_idx: int = max(
            (k for k, v in moving_map_inv.items() if v >= 251),
            default=max(moving_map_inv.keys()),
        )
        model_params = self.ARCH.copy()
        model_params["train"] = {**model_params.get("train", {}), "batch_size": 1}
        self._model = MFMOS(
            nclasses=nclasses,
            movable_nclasses=movable_nclasses,
            params=model_params,
            num_batch=1,
        )

        log.info("Loading MF-MOS checkpoint from %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Upstream checkpoint was saved DataParallel-wrapped — strip "module."
        # prefix so keys line up with the unwrapped model.
        if state and all(k.startswith("module.") for k in state.keys()):
            state = {k[len("module.") :]: v for k, v in state.items()}
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing:
            log.warning("MF-MOS checkpoint missing keys: %s", missing[:5])
        if unexpected:
            log.warning("MF-MOS checkpoint unexpected keys: %s", unexpected[:5])

        self._model.to(self.device).eval()
        log.info(
            "MF-MOS model loaded on %s (native %dx%d, n_input_scans=%d)",
            device,
            self.H_model,
            self.W_model,
            self.n_input_scans,
        )

    def infer(
        self,
        range_image: np.ndarray,
        residual_images: list[np.ndarray],
    ) -> np.ndarray:
        """Run inference on one scan.

        Args:
            range_image: (5, H, W) float32 — channels [range, x, y, z, intensity].
                         Empty pixels should have range=-1.0.
            residual_images: list of (H, W) float32 residual range images.
                             Length may differ from n_input_scans (auto-padded/truncated).

        Returns:
            (H, W) float32 — moving-class probability in [0, 1], at input resolution.
        """
        import torch
        import torch.nn.functional as F

        H_sensor, W_sensor = range_image.shape[1], range_image.shape[2]

        means = np.array(self.img_means, dtype=np.float32).reshape(5, 1, 1)
        stds = np.array(self.img_stds, dtype=np.float32).reshape(5, 1, 1)
        # Only normalise valid pixels — empty pixels stay at 0 (sentinel).
        valid = range_image[0:1] >= 0
        norm_ri = np.where(valid, (range_image - means) / (stds + 1e-8), 0.0).astype(
            np.float32
        )

        n_res = self.n_input_scans
        padded_residuals: list[np.ndarray] = []
        for k in range(n_res):
            if k < len(residual_images):
                padded_residuals.append(residual_images[k].astype(np.float32))
            else:
                padded_residuals.append(
                    np.zeros((H_sensor, W_sensor), dtype=np.float32)
                )
        res_stack = np.stack(padded_residuals, axis=0)
        res_stack = (res_stack - self.res_mean) / (self.res_std + 1e-8)

        x_np = np.concatenate([norm_ri, res_stack], axis=0)  # (5+n_res, H, W)
        x_t = torch.from_numpy(x_np).unsqueeze(0)  # (1, 5+n_res, H, W)

        if H_sensor != self.H_model or W_sensor != self.W_model:
            x_t = F.interpolate(
                x_t,
                size=(self.H_model, self.W_model),
                mode="bilinear",
                align_corners=False,
            )

        x_t = x_t.to(self.device)

        with torch.inference_mode():
            # logits: (1, nclasses, H_model, W_model) — softmax already applied.
            logits, _, _, _ = self._model(x_t)
            score_t = logits[0, self._moving_class_idx, :, :]

        score_np = score_t.cpu().numpy().astype(np.float32)

        if H_sensor != self.H_model or W_sensor != self.W_model:
            score_t2 = torch.from_numpy(score_np).unsqueeze(0).unsqueeze(0)
            score_t2 = F.interpolate(
                score_t2,
                size=(H_sensor, W_sensor),
                mode="bilinear",
                align_corners=False,
            )
            score_np = score_t2[0, 0].numpy().astype(np.float32)

        return score_np
