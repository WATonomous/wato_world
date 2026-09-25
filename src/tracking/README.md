# tracking

4D tracking (3D Kalman filter, masklet association, DINOv2 ReID).

See `wato_world/README.md` and the architecture doc for context.

## Motion model and association

Build on `wato_common.tracking` rather than a new motion model: yaw-only box
fitting and 3D IoU (`boxes.py`), a constant-velocity Kalman filter
(`kalman.py`), and Chen et al.'s association cost with Hungarian assignment,
gating and a deactivated-track buffer (`association.py`). lidar_preprocessing's
Step F already scores cluster motion with these primitives, so the two stages
agree on what "moved" means. Add appearance (DINOv2 ReID) and class on top.

lidar_preprocessing's `motion_clusters.parquet` `track_hint_id` is chunk-local
and geometry-only — useful as a hint, never as an identity.
