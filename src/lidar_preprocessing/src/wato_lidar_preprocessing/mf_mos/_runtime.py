"""Lazy torch + MF-MOS imports for inference.

This module is the ONLY place in the package that imports torch or the
MF-MOS vendored code.  It is never imported at package load time — only when
mf_mos.process_chunk actually calls _load_model().  This keeps test
collection and the CPU-only pipeline path free from torch startup cost.

The MF-MOS submodule lives at:
  src/lidar_preprocessing/third_party/MF-MOS/

and is added to sys.path on first import of this module.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# sys.path injection — adds the submodule once, idempotently.
# ---------------------------------------------------------------------------

_MFMOS_ROOT = str(
    Path(__file__).parent.parent.parent.parent  # src/lidar_preprocessing/
    / "third_party"
    / "MF-MOS"
)

if _MFMOS_ROOT not in sys.path:
    sys.path.insert(0, _MFMOS_ROOT)


# ---------------------------------------------------------------------------
# MFMosModel adapter
# ---------------------------------------------------------------------------


class MFMosModel:
    """Thin wrapper around the upstream MFMOS nn.Module.

    Handles:
    - Loading arch/data YAML configs
    - Building the model at its training resolution (e.g. 64×2048 for KITTI)
    - Loading the checkpoint state dict
    - Normalising inputs using img_means/img_stds from the data config
    - Resizing sensor-resolution inputs to model resolution and back
    - Zero-padding/truncating residual channels to match n_input_scans

    infer() is the only public method; everything else is internal.
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
        self.img_stds: list[float] = sensor["img_stds"]    # 5 values
        self.res_mean: float = float(sensor.get("res_mean", 0.0))
        self.res_std: float = float(sensor.get("res_std", 1.0))

        self.device = torch.device(device)

        # Build model. num_batch must match what MetaKernel was trained with.
        # At inference we use batch_size=1 regardless of training batch_size.
        from modules.MFMOS import MFMOS  # type: ignore[import]

        moving_map_inv = self.DATA.get("moving_learning_map_inv", {0: 0, 1: 251})
        movable_map_inv = self.DATA.get("movable_learning_map_inv", {0: 0, 1: 250})
        nclasses = len(moving_map_inv)
        movable_nclasses = len(movable_map_inv)
        # Moving class is the key whose value is a "moving" semantic label (>=251).
        # Falls back to the highest class index if none found (2-class models).
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

        # Load weights — strict=False to tolerate minor arch divergences.
        log.info("Loading MF-MOS checkpoint from %s", checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Checkpoints may be plain state dicts or wrapped in a dict.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Upstream checkpoint was saved from a DataParallel-wrapped model,
        # so every key is prefixed with "module.". Strip it before loading.
        if state and all(k.startswith("module.") for k in state.keys()):
            state = {k[len("module."):]: v for k, v in state.items()}
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

    @property
    def model_resolution(self) -> tuple[int, int]:
        return self.H_model, self.W_model

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

        # Normalise range image channels using training statistics.
        means = np.array(self.img_means, dtype=np.float32).reshape(5, 1, 1)
        stds = np.array(self.img_stds, dtype=np.float32).reshape(5, 1, 1)
        # Only normalise valid pixels (range >= 0); leave empty pixels at 0.
        valid = range_image[0:1] >= 0  # (1, H, W) bool
        norm_ri = np.where(valid, (range_image - means) / (stds + 1e-8), 0.0).astype(
            np.float32
        )

        # Pad or truncate residuals to n_input_scans channels.
        n_res = self.n_input_scans
        padded_residuals: list[np.ndarray] = []
        for k in range(n_res):
            if k < len(residual_images):
                padded_residuals.append(residual_images[k].astype(np.float32))
            else:
                padded_residuals.append(
                    np.zeros((H_sensor, W_sensor), dtype=np.float32)
                )
        res_stack = np.stack(padded_residuals, axis=0)  # (n_res, H, W)
        res_stack = (res_stack - self.res_mean) / (self.res_std + 1e-8)

        # Concatenate to (5 + n_res, H_sensor, W_sensor) then add batch dim.
        x_np = np.concatenate([norm_ri, res_stack], axis=0)  # (5+n_res, H, W)
        x_t = torch.from_numpy(x_np).unsqueeze(0)  # (1, 5+n_res, H, W)

        # Resize to model's training resolution if different from sensor resolution.
        if H_sensor != self.H_model or W_sensor != self.W_model:
            x_t = F.interpolate(
                x_t,
                size=(self.H_model, self.W_model),
                mode="bilinear",
                align_corners=False,
            )

        x_t = x_t.to(self.device)

        with torch.inference_mode():
            logits, _, _, _ = self._model(x_t)
            # logits: (1, nclasses, H_model, W_model), softmax already applied.
            # _moving_class_idx is derived from moving_learning_map_inv at init time.
            score_t = logits[0, self._moving_class_idx, :, :]  # (H_model, W_model)

        score_np = score_t.cpu().numpy().astype(np.float32)

        # Resize score back to sensor resolution.
        if H_sensor != self.H_model or W_sensor != self.W_model:
            score_t2 = torch.from_numpy(score_np).unsqueeze(0).unsqueeze(0)
            score_t2 = F.interpolate(
                score_t2,
                size=(H_sensor, W_sensor),
                mode="bilinear",
                align_corners=False,
            )
            score_np = score_t2[0, 0].numpy().astype(np.float32)

        return score_np  # (H_sensor, W_sensor) float32
