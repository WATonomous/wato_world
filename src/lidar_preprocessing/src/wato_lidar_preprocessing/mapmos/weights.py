"""MapMOS checkpoint resolution + validation.

The internal voxel size is locked to the value recorded in the
checkpoint metadata (plan non-negotiable #10). It is NOT a user knob in
ComponentConfig — changing it silently breaks the pretrained MinkUNet.

torch is imported lazily so the geometry-only path keeps working when
the with_mapmos Docker stage isn't installed (plan non-negotiable #18).
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Voxel size the pretrained PRBonn MapMOS MinkUNet was trained at.
# TODO(V3): verify against PRBonn config files at the pinned commit and
# update if needed. Locked in code so it can't drift via YAML.
EXPECTED_VOXEL_SIZE_M: float = 0.1


def resolve_weights(weights_path: str) -> str:
    """Return weights_path if the file exists; otherwise raise a clear error."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"MapMOS weights not found at {weights_path!r}. "
            "Run scripts/fetch_mapmos_weights.sh on the host."
        )
    return weights_path


def _extract_ckpt_voxel_size(ckpt: dict[str, Any]) -> float | None:
    """Pull the recorded voxel size out of a Lightning-style or plain dict."""
    if "voxel_size_m" in ckpt:
        return float(ckpt["voxel_size_m"])
    hparams = ckpt.get("hparams") or ckpt.get("hyper_parameters") or {}
    if isinstance(hparams, dict):
        for key in ("voxel_size", "voxel_size_m"):
            if key in hparams:
                return float(hparams[key])
    return None


def load_and_validate(weights_path: str, device: str):
    """Load the MapMOS checkpoint and return (model, ckpt_voxel_size).

    Hard-fails when the recorded voxel size doesn't match
    EXPECTED_VOXEL_SIZE_M. Logs a warning (and assumes the expected value)
    when the checkpoint doesn't record voxel size at all.

    Load idiom verified against PRBonn/MapMOS/src/mapmos/pipeline.py:68-74
    @ commit 8947300698c61257ddb1e1e9f927382f0c0a0bac:
      - Lightning checkpoint stores state_dict under the "state_dict" key.
      - Keys are prefixed with "mos." because training_module.py:46 stores
        the network as `self.mos = MapMOSNet(...)`. We strip the prefix
        so keys match the bare MapMOSNet class layout.
      - .freeze() disables gradients (LightningModule method).
    """
    # Lazy torch import — geometry-only path must not require torch.
    import torch

    from mapmos.mapmos_net import MapMOSNet

    resolve_weights(weights_path)
    ckpt = torch.load(weights_path, map_location="cpu")
    ckpt_voxel_size = _extract_ckpt_voxel_size(ckpt)
    if ckpt_voxel_size is None:
        log.warning(
            "MapMOS checkpoint has no recorded voxel_size — assuming %.3fm",
            EXPECTED_VOXEL_SIZE_M,
        )
        ckpt_voxel_size = EXPECTED_VOXEL_SIZE_M
    if abs(ckpt_voxel_size - EXPECTED_VOXEL_SIZE_M) > 1e-4:
        raise ValueError(
            f"MapMOS checkpoint voxel_size={ckpt_voxel_size} != expected "
            f"{EXPECTED_VOXEL_SIZE_M}. Wrong weights file or wrong model version."
        )

    # Strip the "mos." prefix from saved keys — see source citation above.
    raw_state_dict = ckpt["state_dict"]
    state_dict = {
        k.replace("mos.", "", 1): v
        for k, v in raw_state_dict.items()
        if k.startswith("mos.")
    }
    if not state_dict:
        # Fallback for checkpoints saved without the "mos." prefix
        # (e.g. someone saved a bare MapMOSNet directly).
        state_dict = raw_state_dict

    model = MapMOSNet(voxel_size=ckpt_voxel_size)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    if hasattr(model, "freeze"):
        model.freeze()  # LightningModule.freeze() disables grad
    return model, ckpt_voxel_size
