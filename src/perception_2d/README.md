# perception_2d

2D perception pass (Florence-2 + SAM 3.1 + Depth Anything V2 + DINOv2
appearance embeddings).

Per camera, SAM 3.1's multiplex concept-video predictor detects, segments, and
tracks every instance of each concept across the whole frame stream into
masklets (persistent object ids), with DINOv2 appearance embeddings extracted
every k frames (for downstream re-identification — not used for tracking here).
The concept vocabulary comes from one of two sources, selected by
`discovery.backend`:

- `fixed` (default) — a closed-set class list (`discovery.fixed_classes`,
  e.g. COCO classes); falls back to the prompts.yaml taxonomy when empty.
- `florence2` — open-vocabulary noun phrases discovered per frame by Florence-2,
  pooled into one concept set per camera stream.

Depth Anything V2 + a LiDAR-anchored affine fit produces a metric depth map per
frame. Model loading is fail-loud: a missing SAM 3.1 / Depth Anything / Florence-2
raises rather than emitting degraded placeholder output.

Cross-camera identity merging is **not** done here — it belongs to the
downstream `tracking` component (3D/4D association gated by the DINOv2 features
persisted on each masklet). `global_object_id` is left null for `tracking`.

See `wato_world/README.md` and the architecture doc for context.
