"""Shared SAM 3.1 multiplex video predictor loader (cache).

SAM 3.1's multiplex tracker is NOT available through HuggingFace transformers —
`facebook/sam3.1` hosts checkpoints only ("there is no Hugging Face Transformers
integration … visit the SAM 3 GitHub repository"). It is loaded through Meta's
official `sam3` package (facebookresearch/sam3) via
`build_sam3_multiplex_video_predictor`, which loads `sam3.1_multiplex.pt`
directly (the checkpoint is built for that code).

The predictor is heavy and GPU-resident; this module caches one instance so the
concept tracker shares it across cameras and chunks.

Lazy-imports `sam3` so this module can be imported without it installed; callers
treat a None return as "SAM 3.1 unavailable".
"""

from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# (version, use_fa3) → predictor
_cache: dict[tuple, object] = {}

# Sentinel meaning "MultiplexState.remove_objects was not called this removal"
# (the demo's remove_objects early-returns before touching buckets when every
# object is gone), so the trim-stub restore knows not to re-slice obj_ptr.
_NO_BUCKETS = object()


def get_sam3_predictor(
    version: str = "sam3.1",
    use_fa3: bool = False,
    offload_output_to_cpu: bool = True,
    trim_past_non_cond_mem: bool = True,
    forward_backbone_per_frame: bool = True,
    multiplex_count: int = 8,
    max_num_objects: int = 8,
    image_size: Optional[int] = None,
    postprocess_batch_size: Optional[int] = None,
    batched_grounding_batch_size: Optional[int] = None,
) -> Optional[object]:
    """Return a cached Sam3MultiplexVideoPredictor, or None if unavailable.

    None means the `sam3` package or its checkpoint isn't installed/fetched.
    The pipeline treats None as a hard error for the chunk (empty output + log);
    there is no hand-rolled tracking fallback.

    Memory knobs (all aimed at not OOMing a 24 GB card on long, high-res clips):
      offload_output_to_cpu / trim_past_non_cond_mem / forward_backbone_per_frame
        — flip SAM 3.1's eval memory switches on the tracker
        (see _apply_long_video_memory_savers).
      multiplex_count / max_num_objects — how many parallel hypotheses / objects
        the multiplex detector+tracker carries.

      The next three are the ones that actually move the needle on a 24 GB card.
      They are NOT parameters of build_sam3_multiplex_video_predictor (it hard-
      codes image_size=1008, postprocess_batch_size=16, batched_grounding_batch_
      size=16 when it assembles Sam3MultiplexTrackingWithInteractivity), so we
      set them as attributes on the built `predictor.model` afterwards (see
      _apply_inference_memory_knobs). None = leave the upstream build value.

      batched_grounding_batch_size — THE big one. The detector grounds this many
        frames through its ViT backbone in a single batch; at the built-in 16 and
        image_size 1008 that one allocation is ~20 GB and OOMs a 24 GB card on
        the *first* batch (the propagation bar dies at frame 16). Lower to 1–2.
      postprocess_batch_size — frames accumulated before postprocessing runs on
        them together; the buffer is postprocess_batch_size * max_num_objects.
        The class default is 1 ("set to 1 to disable batching"); the builder
        forces 16. Lower to 1 to postprocess per frame.
      image_size — inference resolution each frame is resized to before the
        backbone. The backbone uses real-valued RoPE, so it tolerates a smaller
        size; memory falls ~quadratically. Lower (e.g. 768) at the cost of
        small-object recall. None keeps the built-in 1008.
    """
    key = (
        version,
        use_fa3,
        offload_output_to_cpu,
        trim_past_non_cond_mem,
        forward_backbone_per_frame,
        multiplex_count,
        max_num_objects,
        image_size,
        postprocess_batch_size,
        batched_grounding_batch_size,
    )
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        from sam3.model_builder import (
            build_sam3_multiplex_video_predictor,
            download_ckpt_from_hf,
        )

        log.info(
            "SAM 3.1: building multiplex video predictor "
            "(use_fa3=%s, multiplex_count=%d, max_num_objects=%d, image_size=%s, "
            "postprocess_batch_size=%s, batched_grounding_batch_size=%s) …",
            use_fa3,
            multiplex_count,
            max_num_objects,
            image_size,
            postprocess_batch_size,
            batched_grounding_batch_size,
        )
        t0 = time.perf_counter()
        ckpt = download_ckpt_from_hf(version=version)
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=ckpt,
            use_fa3=use_fa3,
            multiplex_count=multiplex_count,
            max_num_objects=max_num_objects,
            async_loading_frames=False,  # we hand it in-memory PIL frames
        )
        # The builder bakes image_size=1008 / postprocess_batch_size=16 /
        # batched_grounding_batch_size=16 (sized for 80 GB cards). They are plain
        # attributes on the assembled model, so override them post-build to fit a
        # 24 GB card — this is what actually prevents the OOM, not the long-video
        # switches below.
        _apply_inference_memory_knobs(
            predictor,
            image_size=image_size,
            postprocess_batch_size=postprocess_batch_size,
            batched_grounding_batch_size=batched_grounding_batch_size,
        )
        # trim_past_non_cond_mem is the only lever that bounds the maskmem that
        # accumulates over a long clip (multiplex_count is fixed at 16 by the
        # checkpoint, so it can't be lowered). But sam3's trim path has TWO bugs
        # that KeyError mid-clip — _patch_trim_keyerror (the 'multistep_point_
        # inputs' read in the trim itself) and _patch_remove_objects_keyerror
        # (the 'maskmem_features' read when the multiplex tracker later drops a
        # dead object). Only enable trim if BOTH patches are in place; otherwise
        # a long clip either OOMs (trim off) or aborts at the first removal.
        trim_ok = (
            _patch_trim_keyerror() and _patch_remove_objects_keyerror()
            if trim_past_non_cond_mem
            else False
        )
        _apply_long_video_memory_savers(
            predictor,
            offload_output_to_cpu=offload_output_to_cpu,
            trim_past_non_cond_mem=trim_past_non_cond_mem and trim_ok,
            forward_backbone_per_frame=forward_backbone_per_frame,
        )
        _cache[key] = predictor
        log.info(
            "SAM 3.1 predictor ready in %.1fs (checkpoint=%s)",
            time.perf_counter() - t0,
            ckpt,
        )
        return predictor
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "SAM 3.1 multiplex predictor unavailable (%s) — install the `sam3` "
            "package and fetch facebook/sam3.1 into the HF cache.",
            exc,
        )
        return None


def _apply_inference_memory_knobs(
    predictor: object,
    *,
    image_size: Optional[int],
    postprocess_batch_size: Optional[int],
    batched_grounding_batch_size: Optional[int],
) -> None:
    """Override the build-time memory knobs on the assembled multiplex model.

    build_sam3_multiplex_video_predictor hardcodes image_size=1008,
    postprocess_batch_size=16 and batched_grounding_batch_size=16 (sized for an
    80 GB card) when it constructs Sam3MultiplexTrackingWithInteractivity, and
    exposes none of them as builder kwargs. They are plain instance attributes,
    so we set them directly on predictor.model afterwards. A None leaves the
    upstream value. We only set an attribute that already exists (so a sam3 rev
    that renamed one is a logged no-op, not an AttributeError that nukes the
    whole predictor).

    batched_grounding_batch_size is the one that prevents the OOM: at 16 the
    detector grounds 16 frames at 1008px in a single ViT pass (~20 GB), which
    OOMs the first batch on a 24 GB card (the propagation bar dies at frame 16).
    """
    model = getattr(predictor, "model", None)
    if model is None:
        log.warning("SAM 3.1: predictor has no .model — cannot apply memory knobs.")
        return
    requested = {
        "image_size": image_size,
        "postprocess_batch_size": postprocess_batch_size,
        "batched_grounding_batch_size": batched_grounding_batch_size,
    }
    for attr, val in requested.items():
        if val is None:
            continue
        if hasattr(model, attr):
            old = getattr(model, attr)
            setattr(model, attr, val)
            log.info("SAM 3.1: set model.%s %s -> %s", attr, old, val)
        else:
            log.warning(
                "SAM 3.1: model has no attribute %s — cannot override it "
                "(sam3 internals may have changed); leaving the build default.",
                attr,
            )


def _apply_long_video_memory_savers(
    predictor: object,
    *,
    offload_output_to_cpu: bool,
    trim_past_non_cond_mem: bool,
    forward_backbone_per_frame: bool = True,
) -> dict[str, int]:
    """Flip SAM 3.1's long-video / large-resolution eval memory switches on
    every tracker submodule that exposes them:

      forward_backbone_per_frame → forward_backbone_per_frame_for_eval: compute
        the image backbone one frame at a time instead of over all frames at
        once. Upstream's *first* recommended remedy "to avoid backbone OOM
        errors on very long videos". Cuts both the backbone activation spike
        and the resident feature cache.
      offload_output_to_cpu → offload_output_to_cpu_for_eval: page each frame's
        pred_masks / maskmem features to host RAM (paged back on demand).
      trim_past_non_cond_mem → trim_past_non_cond_mem_for_eval: drop the maskmem
        of frames that have slid past the num_maskmem attention window (obj_ptr
        kept). Quality-neutral when only the first frame is prompted; upstream
        requires num_frames_to_correct_for_eval <= 1, so we gate on that.

    NOTE: these reduce baseline/accumulation and the backbone spike. They do
    NOT shrink the per-step detector/tracker attention transient — for that,
    lower multiplex_count / max_num_objects (or image_size) at build time.

    Walks predictor.model.modules() so we don't hard-code the (version-fragile)
    attribute path down to the VideoTrackingMultiplex instance. Returns a
    {flag: module_count} map (zeros + a warning if the attrs are absent, which
    would mean the switches silently did nothing).
    """
    model = getattr(predictor, "model", None)
    modules = list(model.modules()) if hasattr(model, "modules") else []
    counts = {
        "forward_backbone_per_frame_for_eval": 0,
        "offload_output_to_cpu_for_eval": 0,
        "trim_past_non_cond_mem_for_eval": 0,
    }
    skipped_trim = 0
    for m in modules:
        if forward_backbone_per_frame and hasattr(
            m, "forward_backbone_per_frame_for_eval"
        ):
            m.forward_backbone_per_frame_for_eval = True
            counts["forward_backbone_per_frame_for_eval"] += 1
        if offload_output_to_cpu and hasattr(m, "offload_output_to_cpu_for_eval"):
            m.offload_output_to_cpu_for_eval = True
            counts["offload_output_to_cpu_for_eval"] += 1
        if trim_past_non_cond_mem and hasattr(m, "trim_past_non_cond_mem_for_eval"):
            if getattr(m, "num_frames_to_correct_for_eval", 1) <= 1:
                m.trim_past_non_cond_mem_for_eval = True
                counts["trim_past_non_cond_mem_for_eval"] += 1
            else:
                skipped_trim += 1
    log.info("SAM 3.1 long-video memory savers applied: %s", counts)
    if skipped_trim:
        log.warning(
            "SAM 3.1: skipped trim_past_non_cond_mem on %d module(s) configured "
            "for multi-frame correction (num_frames_to_correct_for_eval > 1)",
            skipped_trim,
        )
    requested = {
        "forward_backbone_per_frame_for_eval": forward_backbone_per_frame,
        "offload_output_to_cpu_for_eval": offload_output_to_cpu,
        "trim_past_non_cond_mem_for_eval": trim_past_non_cond_mem,
    }
    for flag, on in requested.items():
        if on and counts[flag] == 0 and flag != "trim_past_non_cond_mem_for_eval":
            log.warning(
                "SAM 3.1: requested %s but no module exposes it — switch had no "
                "effect (sam3 internals may have changed)",
                flag,
            )
    return counts


def _backfill_multistep_key(output_dict: object) -> int:
    """setdefault 'multistep_point_inputs'=None on every stored frame output in
    `output_dict` (cond + non-cond), returning the number of dicts touched.

    The key is write-only during sam3 inference (only ever assigned, never read),
    so this is behaviourally inert — its sole purpose is to stop the trim path's
    unconditional `past_out["multistep_point_inputs"]` from raising KeyError when
    the past frame is a compacted / add_prompt output that never carried it.
    """
    if not isinstance(output_dict, dict):
        return 0
    touched = 0
    for storage in ("cond_frame_outputs", "non_cond_frame_outputs"):
        bucket = output_dict.get(storage)
        if not isinstance(bucket, dict):
            continue
        for out in bucket.values():
            if isinstance(out, dict) and "multistep_point_inputs" not in out:
                out["multistep_point_inputs"] = None
                touched += 1
    return touched


def _patch_trim_keyerror() -> bool:
    """Make sam3's trim_past_non_cond_mem path tolerate a missing
    'multistep_point_inputs' key. Returns True if the patch is in place.

    Upstream bug (sam3 @8e451d5, pinned in docker/perception_2d.Dockerfile):
    VideoTrackingMultiplex._trim_output_and_memory calls a nested _trim_past_out
    that does `past_out["multistep_point_inputs"]` UNCONDITIONALLY
    (video_tracking_multiplex.py:2494). But the multiplex demo stores *compacted*
    per-frame outputs — and the add_prompt conditioning frame — WITHOUT that key
    (video_tracking_multiplex_demo.py:_run_single_frame_inference). So the moment
    trim_past_non_cond_mem_for_eval is on, propagation dies mid-clip with
    KeyError('multistep_point_inputs') at a data-dependent frame. Verified with a
    full traceback on the installed package.

    The key is WRITE-ONLY during inference (grep across sam3: it is only ever
    assigned, never read — it exists for training losses), so defaulting it to
    None is behaviourally inert. _trim_past_out is a nested local we can't patch
    directly, so we wrap the enclosing method: before it runs, inject the key
    into every stored frame output that lacks it. Idempotent; applied once per
    process. Without this, trim_past_non_cond_mem must stay off and long
    high-res clips OOM (~frame 80 at 1024x1280 on a 24 GB card).
    """
    try:
        from sam3.model.video_tracking_multiplex import VideoTrackingMultiplex
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "SAM 3.1: could not import VideoTrackingMultiplex to patch the trim "
            "KeyError (%s) — leaving trim_past_non_cond_mem OFF.",
            exc,
        )
        return False
    if getattr(VideoTrackingMultiplex, "_wato_trim_patched", False):
        return True

    _orig = VideoTrackingMultiplex._trim_output_and_memory

    def _patched(self, *args, **kwargs):
        output_dict = kwargs.get("output_dict")
        if output_dict is None and len(args) >= 2:
            output_dict = args[1]
        if (
            getattr(self, "trim_past_non_cond_mem_for_eval", False)
            and not self.training
        ):
            _backfill_multistep_key(output_dict)
        return _orig(self, *args, **kwargs)

    VideoTrackingMultiplex._trim_output_and_memory = _patched
    VideoTrackingMultiplex._wato_trim_patched = True
    log.info(
        "SAM 3.1: patched VideoTrackingMultiplex._trim_output_and_memory to "
        "tolerate missing 'multistep_point_inputs' (upstream trim KeyError)."
    )
    return True


def _pop_trimmed_noncond(output_dict: object) -> dict:
    """Lift trim-stub entries out of ``non_cond_frame_outputs``, returning them.

    Once trim_past_non_cond_mem_for_eval is on, _trim_past_out reduces each past
    non-conditioning frame to a pointer-only stub (obj_ptr + pred_masks + scores)
    that no longer carries 'maskmem_features' / 'maskmem_pos_enc' /
    'local_obj_id_to_idx'. Upstream's remove_objects → _slice_state assumes every
    non-cond frame is still a full output and unconditionally indexes
    ``out["maskmem_features"][buckets_to_keep]``, so it KeyErrors on the first
    stub the moment the multiplex tracker drops a dead object. We pop the stubs
    aside so _slice_state only ever sees full frames, then restore them.

    A stub is identified purely by the absence of 'maskmem_features' — which also
    correctly catches a (rare) empty frame that never ran the memory encoder, the
    other case _slice_state would choke on. Returns {frame_idx: out}; empty when
    there are no stubs or the structure isn't what we expect (then the original
    runs unchanged and any genuine error surfaces normally).
    """
    if not isinstance(output_dict, dict):
        return {}
    noncond = output_dict.get("non_cond_frame_outputs")
    if not isinstance(noncond, dict):
        return {}
    popped: dict = {}
    for frame_idx, out in list(noncond.items()):
        if isinstance(out, dict) and "maskmem_features" not in out:
            popped[frame_idx] = out
            del noncond[frame_idx]
    return popped


def _restore_trimmed_noncond(
    output_dict: object, popped: dict, buckets_to_keep: object
) -> int:
    """Re-insert the stubs popped by _pop_trimmed_noncond into the output dict.

    ``buckets_to_keep`` is the list[int] of surviving multiplex buckets returned
    by MultiplexState.remove_objects (captured in _patch_remove_objects_keyerror),
    or _NO_BUCKETS when the demo's remove_objects early-returned without touching
    buckets (every object removed → the inference state is discarded anyway). When
    it is a real list we slice each stub's obj_ptr by it, reproducing the one
    bucket-dim slice _slice_state would have applied to a stub (obj_ptr is the
    only bucketed tensor a stub still holds), so its bucket dim stays consistent
    with the frames upstream just sliced. Returns the count of stubs re-sliced.
    """
    if not popped:
        return 0
    noncond = (
        output_dict.get("non_cond_frame_outputs")
        if isinstance(output_dict, dict)
        else None
    )
    apply = buckets_to_keep is not _NO_BUCKETS and buckets_to_keep is not None
    sliced = 0
    for frame_idx, out in popped.items():
        if apply and isinstance(out, dict) and "obj_ptr" in out:
            try:
                out["obj_ptr"] = out["obj_ptr"][buckets_to_keep]
                sliced += 1
            except Exception as exc:  # noqa: BLE001
                # A stale obj_ptr only bites later if a whole bucket was dropped
                # (rare); never lose the stub over it.
                log.warning(
                    "SAM 3.1: could not re-slice a trimmed frame's obj_ptr to the "
                    "surviving buckets (%s) — leaving it; bucket removal is rare.",
                    exc,
                )
        if isinstance(noncond, dict):
            noncond[frame_idx] = out
    return sliced


def _patch_remove_objects_keyerror() -> bool:
    """Make sam3's multiplex object-removal tolerate trim-stub frames. Returns
    True if the patch is in place.

    Upstream bug (sam3 @8e451d5, pinned in docker/perception_2d.Dockerfile):
    VideoTrackingMultiplexDemo.remove_objects rebuilds the packed per-bucket
    tensors via a nested _slice_state that does
    ``out["maskmem_features"][buckets_to_keep]`` for EVERY non-cond frame
    (video_tracking_multiplex_demo.py:3096). But with trim_past_non_cond_mem on,
    _trim_past_out has already replaced past non-cond frames with pointer-only
    stubs that dropped 'maskmem_features'. So the first time the multiplex tracker
    removes a dead object (which it does routinely, mid-clip), _slice_state
    KeyErrors on 'maskmem_features' and propagation dies — observed at frame 61 /
    120 on real nuScenes clips, salvaging only a handful of tracks per camera.

    _slice_state is a nested local we can't patch directly (same constraint as the
    trim KeyError), so we wrap the enclosing remove_objects: pop the maskmem-less
    stubs out of non_cond_frame_outputs before it runs (so _slice_state only sees
    full frames) and restore them after, slicing each stub's obj_ptr to the
    surviving buckets so its bucket dim still lines up. To learn which buckets
    survived we also wrap MultiplexState.remove_objects (the call that computes
    buckets_to_keep) to stash its return on the instance. Idempotent; applied once
    per process. Without this, trim_past_non_cond_mem must stay off (→ long
    high-res clips OOM) or tracking aborts at the first object removal.
    """
    try:
        from sam3.model.multiplex_utils import MultiplexState
        from sam3.model.video_tracking_multiplex_demo import (
            VideoTrackingMultiplexDemo,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "SAM 3.1: could not import the multiplex classes to patch the "
            "object-removal KeyError (%s) — leaving trim_past_non_cond_mem OFF.",
            exc,
        )
        return False

    # 1. Capture buckets_to_keep: MultiplexState.remove_objects returns the list
    #    of surviving bucket indices. Stash it on the instance so the wrapper
    #    below can re-slice the stubs it set aside.
    if not getattr(MultiplexState, "_wato_capture_patched", False):
        _ms_orig = MultiplexState.remove_objects

        def _ms_patched(self, object_indices, strict=True):
            result = _ms_orig(self, object_indices, strict=strict)
            self._wato_last_buckets_to_keep = result
            return result

        MultiplexState.remove_objects = _ms_patched
        MultiplexState._wato_capture_patched = True

    # 2. Quarantine the trim stubs around the demo's remove_objects.
    if getattr(VideoTrackingMultiplexDemo, "_wato_remove_patched", False):
        return True

    _orig = VideoTrackingMultiplexDemo.remove_objects

    def _patched(self, *args, **kwargs):
        inference_state = args[0] if args else kwargs.get("inference_state")
        state = inference_state if isinstance(inference_state, dict) else None
        output_dict = state.get("output_dict") if state is not None else None
        ms = state.get("multiplex_state") if state is not None else None
        popped = _pop_trimmed_noncond(output_dict)
        if ms is not None:
            try:
                # Reset so an early-return (no bucket call) is distinguishable
                # from a real removal that kept buckets.
                ms._wato_last_buckets_to_keep = _NO_BUCKETS
            except Exception:  # noqa: BLE001
                ms = None  # can't stash on it → treat as no capture
        try:
            return _orig(self, *args, **kwargs)
        finally:
            btk = (
                getattr(ms, "_wato_last_buckets_to_keep", _NO_BUCKETS)
                if ms is not None
                else _NO_BUCKETS
            )
            _restore_trimmed_noncond(output_dict, popped, btk)

    VideoTrackingMultiplexDemo.remove_objects = _patched
    VideoTrackingMultiplexDemo._wato_remove_patched = True
    log.info(
        "SAM 3.1: patched VideoTrackingMultiplexDemo.remove_objects to tolerate "
        "trimmed (maskmem-less) past frames (upstream object-removal KeyError)."
    )
    return True


def sam3_importable() -> bool:
    """Cheap check that `sam3` + its deps import, WITHOUT building the heavy
    GPU-resident predictor.

    The pipeline calls this before the (expensive) depth pass so a missing
    `sam3` install / dependency fails fast instead of after hours of depth.
    Actually executes the import (not importlib.util.find_spec) so a broken
    transitive dep — e.g. a missing pycocotools — is caught, not just an
    absent top-level package.
    """
    import importlib

    try:
        importlib.import_module("sam3.model_builder")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("SAM 3.1 (`sam3.model_builder`) not importable: %s", exc)
        return False


def release_sam3_predictor() -> None:
    """Drop the cached predictor and free its GPU memory (best-effort).

    Called at the end of a bag so a long-lived process doesn't keep SAM 3.1
    parked in VRAM after the run.
    """
    _cache.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
