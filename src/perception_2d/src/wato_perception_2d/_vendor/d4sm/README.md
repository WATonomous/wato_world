# DAM4SAM author source

Source: https://github.com/alanlukezic/d4sm
Commit: `67d5dfe86d8ae7b557193231c6058a46f5c0c490` (2025-10-19).
This is the multiobject implementation linked by the original DAM4SAM authors.
Apache-2.0 license is included in `LICENSE`; the upstream secondary license is
preserved in `LICENSE_cctorch`. Meta copyright notices are retained.

`sam2_base.py` is byte-identical to the authors' `sam2/modeling/sam2_base.py`
at that commit. That file is **the authors' modified copy of Meta's
`SAM2Base`, not Meta's file**. Against the SAM2 pinned in
`docker/perception_2d.Dockerfile` it:

- defaults `memory_temporal_stride_for_eval` to 5 (Meta: 1). The SAM2.1 Hydra
  config doesn't set it, so this default is where DAM4SAM's stride 5 comes from;
- takes memory positional encodings from the per-object output dict;
- accepts `num_frames=None` when capping object pointers;
- adds `_prepare_memory_conditioned_features_multi`.

It imports the other model submodules unchanged from the installed Meta SAM2.
A Hydra target override loads this class without replacing the installed
`sam2` package. Existing SAM2.1 weights load strictly; no training or
additional checkpoint is needed.

`tracking_wrapper_mot.py` is copied from the repository root with these changes:

1. Add this local-modification notice.
2. Replace VOT toolkit imports and its single rectangle-overlap call with
   `rectangle_overlap.mask_box_overlap`. This preserves VOT semantics for the
   positive integer XYWH boxes produced from masks, including its degenerate
   union branch. No VOT dataset/runner dependency is needed.
3. Accept a shared `sam` model and explicit `device` in the constructor; avoid
   loading another model when one is supplied.
4. Replace hardcoded CUDA transfers/device fields with that device.
5. Return the already-computed full-resolution `masks_out` as `mask_logits`
   alongside the original binary masks. The pipeline needs logits for ownership.
6. Convert predicted IoUs to float32 before NumPy export. The pipeline uses
   bfloat16 autocast; NumPy cannot represent that dtype. Conversion preserves
   the represented values and the author's thresholds.

**Memory selection is unchanged.** Defaults remain 3 RAM slots, 3 DRM slots,
the initial memory, stride 5, last-frame memory, and 16 object pointers. DRM
requires predicted IoU > .8, size ratio [.8, 1.2], the published update spacing,
and alternative-mask union-box IoU <= .7. Largest-component filtering applies
only to alternative masks for distractor selection, not the output object mask.

The adapter in `models/dam4sam.py` supplies one context per object and shares
the authors' prepared image/features once per frame. This avoids the author's
>200-object batching bug. A late birth starts relative frame 0; a correction
replaces that object's context and resets its relative clock; removal frees its
context. These lifecycle choices are application adaptations, not a reproduction
of the paper's ground-truth-first-mask benchmark. Reverse propagation is not
supported (back-fill feeds earlier frames forward in reverse order). The model
and bounded per-object memories stay on its device; frames are read lazily on
CPU. Upstream's offload-state flag is unused and not exposed.

Two adapter parameters change what the memory keeps, from its context subclass,
leaving both copied files as they are; the defaults reproduce the authors' code
(the pipeline's values are in `segmentation.dam4sam`):

- `update_delta` sets the context's `update_delta` attribute (the authors'
  Δ, which spaces both RAM and DRM updates).
- `memory_min_predicted_iou` ports SAMURAI's memory-bank gate: after the next
  frame's memory attention, the previous frame's RAM entry is removed if its
  predicted IoU was at most the gate, so the authors' update then treats the
  new frame as a replacement for it. A frame waiting to be swapped into the DRM
  is left to the authors' swap. Object pointers are not gated.

A thin model view hands each `track_step` output to its context, which records
the chosen mask's predicted IoU and object-score logit for diagnostics.

Do not reformat the two copied upstream files. SHA-256 of the upstream files at
the pinned commit (`sam2_base.py` here must still match the first):

| Upstream path | SHA-256 |
|---|---|
| `sam2/modeling/sam2_base.py` | `3ec3600badfb8465471276e4e9fe0adb9ccfce920cb328dae4da8c0bf014afe6` |
| `tracking_wrapper_mot.py` (before the edits above) | `85f42a0a43a41a10bebac409775b0b67f3e923cd1af9dec10cf9bfb4284bad57` |
