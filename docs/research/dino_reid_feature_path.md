# Current DINOv2 / ReID feature path

> Since 2026-11-12 the descriptor, its storage and its consumer differ from
> what this note inspects: `src/perception_2d/README.md` (pass 3),
> `src/tracking/README.md` ("Appearance"), `docs/tracking_measurements.md`
> ("Appearance").

Read-only evaluation, 2026-11-11. No model run, association experiment, or code /
config change was made. **Features are extracted and persisted; appearance-based
ReID is not connected to either 2D or 3D tracking.**

## Extraction: valid global descriptor, limited object isolation

- The pinned `dinov2_vitl14` backbone is loaded through Torch Hub, cached by
  model/device, set to evaluation mode, and called under `torch.no_grad()`.
  Empty masks return `None`, while missing dependencies/models raise. Sources:
  [embeddings.py:25](../../src/perception_2d/src/wato_perception_2d/models/embeddings.py#L25),
  [embeddings.py:96](../../src/perception_2d/src/wato_perception_2d/models/embeddings.py#L96),
  [model_registry.py:97](../../src/perception_2d/src/wato_perception_2d/model_registry.py#L97).
- Each mask is used **only to find its bounding box**. Its RGB crop retains
  background and other objects inside that rectangle. It is resized to short
  side 224, center-cropped to 224×224, converted to a tensor, and normalized with
  ImageNet mean/std. This is neither foreground masking nor masked patch-token
  pooling. Two masks with the same bounding rectangle produce the same input.
  Sources: [embeddings.py:60](../../src/perception_2d/src/wato_perception_2d/models/embeddings.py#L60),
  [embeddings.py:113](../../src/perception_2d/src/wato_perception_2d/models/embeddings.py#L113).
- `model(batch)` correctly returns the backbone CLS descriptor: the pinned
  official implementation applies LayerNorm and an identity head to the CLS
  token; ViT-L outputs 1,024 dimensions. **LayerNorm is not unit L2
  normalization.** Neither local extraction nor aggregation L2-normalizes
  vectors. Source: [official pinned DINOv2 transformer, lines 91, 154–155,
  238–253, 322–327, 366–379](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/models/vision_transformer.py#L238).
- Input normalization matches official DINOv2 constants. The preprocessing is
  a local choice: official classification evaluation defaults to resize 256,
  bicubic interpolation, then center crop 224; local code uses `Resize(224)`
  without an explicit interpolation argument. This difference is not evidence
  of a broken extractor, but means it does not exactly reproduce that upstream
  evaluation transform. Source: [official pinned transforms, lines 37–45,
  68–81](https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/dinov2/data/transforms.py#L68).

Inference from the crop geometry: a wide car or tall pedestrian loses its ends
to the square center crop; background and a drifting mask can change the
descriptor independently of identity. General visual features are a plausible
starting point, but no same-object/different-object separation on these cameras
has been demonstrated by this code inspection.

## Sampling, aggregation, storage

Both shipped configs set `embeddings.every_k_frames: 1`, so configured runs
embed every recorded forward frame, despite the tracking function's default
argument of five. The pipeline passes the configured value through. Sources:
[WATO config:233](../../src/perception_2d/config/perception_2d.wato.yaml#L233),
[default config:192](../../src/perception_2d/config/perception_2d.yaml#L192),
[pipeline.py:684](../../src/perception_2d/src/wato_perception_2d/pipeline.py#L684).

`_record_frame` batches all present masks when `frame_index_abs % k == 0`.
Masks have passed ego-body removal, island filtering and ended-track handling,
but embedding updates have **no detector-confidence, detector-support,
predicted-IoU, stability, occlusion, size, or sharpness gate**. Tentative masks
are held for writing, yet still reach the embedding batch. Reverse-backfilled
masks are written separately and add no DINO samples. Sources:
[sam2_tracker.py:1036](../../src/perception_2d/src/wato_perception_2d/models/sam2_tracker.py#L1036),
[sam2_tracker.py:1389](../../src/perception_2d/src/wato_perception_2d/models/sam2_tracker.py#L1389),
[sam2_tracker.py:1160](../../src/perception_2d/src/wato_perception_2d/models/sam2_tracker.py#L1160).

All collected descriptors receive equal weight in one arithmetic mean. There
is no feature bank, outlier rejection, sample weighting, or final unit
normalization. **Risk inferred from this rule:** a prolonged drift or identity
switch can dominate the stored descriptor; it cannot represent two identities
separately. Raw-vector averaging also gives differing vector norms influence
over direction. Source:
[sam2_tracker.py:1477](../../src/perception_2d/src/wato_perception_2d/models/sam2_tracker.py#L1477).

The pipeline saves that float32 mean as
`masks_2d/<masklet_id>/dino_feature.npy` and puts its path in the masklet table.
Individual descriptors, their timestamps, quality and sample counts are not
persisted. Sources: [pipeline.py:735](../../src/perception_2d/src/wato_perception_2d/pipeline.py#L735),
[masklets.py:87](../../src/common/src/wato_common/artifacts/masklets.py#L87).

## Actual consumers and missing connection

A repository-wide search for `dino_feature`, `read_dino_feature`,
`extract_dino`, `dino_cosine`, and appearance association found:

| Surface | Current behavior |
| --- | --- |
| Masklet artifact API | Writes arrays and reads path metadata. **No `read_dino_feature` function exists**; `MaskletIndex` reads masks, not embeddings. [masklets.py:149](../../src/common/src/wato_common/artifacts/masklets.py#L149), [masklets.py:196](../../src/common/src/wato_common/artifacts/masklets.py#L196). |
| Array consumer | The round-trip test directly calls `np.load`; no production feature-array reader was found. [test_artifacts_masklets.py:57](../../src/common/tests/test_artifacts_masklets.py#L57). |
| 2D detection association | Uses detection-box to mask-bounding-box IoU. DINO does not accept/reject a match or recover a lost identity. [sam2_tracker.py:167](../../src/perception_2d/src/wato_perception_2d/models/sam2_tracker.py#L167). |
| 3D association | Reads proposals; passes boxes, strong flags, lifted-instance keys and birth permissions to the geometric tracker. [tracking pipeline.py:45](../../src/tracking/src/wato_tracking/pipeline.py#L45), [tracker.py:42](../../src/tracking/src/wato_tracking/tracker.py#L42). |
| Cross-chunk linking | Compares simultaneous world-space centers over the shared span. [stitch.py:38](../../src/tracking/src/wato_tracking/stitch.py#L38). |
| Track artifact field | `dino_feature_path` exists but `_frame()` leaves it at its default `None`. [tracks.py:74](../../src/common/src/wato_common/artifacts/tracks.py#L74), [tracking pipeline.py:61](../../src/tracking/src/wato_tracking/pipeline.py#L61). |
| Legacy `dino_cosine: 1.0` | Appears in root `config/pipeline.yaml`, but current tracking loads its component config and exposes no such weight. It does not enable ReID. [legacy config:31](../../config/pipeline.yaml#L31), [tracking CLI:17](../../src/tracking/src/wato_tracking/cli.py#L17), [tracking config.py:170](../../src/tracking/src/wato_tracking/config.py#L170). |

## Evaluation implications

Changing DINO descriptors alone cannot improve current association metrics:
they have no decision-making consumer. Disabling extraction should leave
algorithmic tracking decisions unchanged, although runtime savings require
measurement. Feature usefulness should be evaluated separately on labelled
same/different identity pairs, especially nearby same-class objects, camera
changes, occlusion and known drift. Compare current CLS crops against
aspect-preserving crops and mask-aware pooling; compare the current mean
against normalized, quality-selected descriptors before choosing any cosine
threshold. These are proposed experiments, not measured improvements or an
implementation plan. Normalizing a stored contaminated mean cannot undo its
mixed identities, and a whole-masklet mean includes future frames, so any later
evaluation must distinguish offline linking from causal online association.
