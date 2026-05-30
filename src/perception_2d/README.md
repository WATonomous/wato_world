# perception_2d

2D perception pass (Florence-2 + SAM 3.1 + Depth Anything V2 + DINOv2
appearance embeddings).

Per camera: discovery proposes class candidates (Florence-2 open-vocabulary
phrases, or a fixed closed-set class list via `discovery.backend: fixed`) →
SAM 3.1 text-prompted segmentation → phrase dedup / 3D-NMS → SAM 3.1 video
tracking into masklets, with DINOv2 appearance embeddings extracted every k
frames (for downstream re-identification — not used for tracking here). Depth
Anything V2 + LiDAR-anchored affine fit produces a metric depth map per frame.

Cross-camera identity merging is **not** done here — it belongs to the
downstream `tracking` component (3D/4D association gated by the DINOv2 features
persisted on each masklet). `global_object_id` is left null for `tracking`.

See `wato_world/README.md` and the architecture doc for context.
