# SLF projection and implementation review

Reviewed 2026-11-03 against primary sources. This note distinguishes reproducible implementation failures from the repository's deliberate rounded-box adaptation.

## Primary reference and method

The paper is **Jianhao Li, Tianyu Sun, Zhongdao Wang, Enze Xie, Bailan Feng, Hongbo Zhang, Ze Yuan, Ke Xu, Jiaheng Liu, and Ping Luo, “Segment, Lift and Fit: Automatic 3D Shape Labeling from 2D Prompts,” ECCV 2024**. The first arXiv submission was July 16, 2024; v2 is July 17. “Li et al. 2023” is incorrect. [Author-submitted record](https://arxiv.org/abs/2407.11382)

| Topic | Verified paper definition |
| --- | --- |
| Shape and pose | Section 3.3: PCA SDF shape, **d = 5**; pose **x, y, z, yaw**, not full SE(3). |
| Rendering | Section 3.4: ray occupancy `1 - product_x sigmoid(-zeta * phi(x))`, not minimum-SDF rendering. |
| Occlusion | Section 3.4, Figure 4, Eq. (2): order masks by median associated LiDAR depth; suppress nearer masks. |
| LiDAR | Eq. (3): absolute SDF plus squared distance to the **first camera-ray surface intersection** for intersecting rays; not Chamfer distance. |
| Ground | Eq. (4): soft penalty on bottom-to-RANSAC-ground distance. |

Section 3.3 describes positive-exterior SDF values, while Section 3.4's rendering explanation assumes positive-interior values. The paper is inconsistent here. The implementation's negative-inside SDF and `sigmoid(-minimum_distance / edge)` are internally consistent. [Official proceedings PDF, Sections 3.3–3.4, pp. 5–8](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11297.pdf)

No author-published implementation was located through the paper record and author website. The author's SLF entry links the paper without a code link. This review therefore does **not** claim verification against official source code. [Tianyu Sun's publication page](https://provemj.github.io/)

## Established failures addressed in this change

- **Incomplete camera geometry.** SLF projection and inverse rays previously used only `fx, fy, cx, cy`, omitting distortion and other intrinsic terms. Forward projection and ray construction now share the calibrated camera model. For a raw pixel, apply the full inverse K, invert distortion in normalized image coordinates, then form a unit camera ray and rotate it into world coordinates. `K^-1` alone cannot invert a distorted raw image. Raw-image `K,D` and rectified-image `R,P` have different contracts. [ROS CameraInfo definition](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/CameraInfo.msg), [OpenCV calibration model](https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html)
- **Different optimized and exported boxes.** Previously, `views.problem` anchored moving tracks to raw observed centers while `apply_slf` rebuilt them through Phase 3 smoothing and another ground lookup. A seven-frame jitter probe showed **0.377142857 m** center error even at unchanged size and zero offset. The implementation now references Phase 3 placement and exports the evaluated per-frame fit centers directly. This is a correctness failure independent of the chosen shape model.
- **Inconsistent mask pixel selection.** Continuous SLF ray coordinates previously sampled masks by flooring while shared lifting selected the nearest pixel. The adapter now uses the shared nearest-pixel convention.
- **World-coordinate precision loss.** Converting large world positions to float32 before subtraction changed fitted dimensions by about **4 cm** under a `1e6 m` translation. Per-track float64 origin subtraction before tensor conversion now preserves the local optimization problem.

These are regression findings from this change's implementation and tests, not claims made by the paper. Relevant code: [views.py](../../src/label_refinement/src/wato_label_refinement/views.py), [slf.py](../../src/label_refinement/src/wato_label_refinement/slf.py), [pipeline.py](../../src/label_refinement/src/wato_label_refinement/pipeline.py).

## Adaptations and remaining limitations

The current model deliberately fits one rounded-box size and shared planar offset per track, with Phase 3 heading and bottom fixed, robust SDF point loss, class size priors, and min-SDF silhouettes. These differ from the method above; they are not evidence of additional projection errors.

The omitted first-intersection constraint leaves a possible front/back surface ambiguity. Points selected only from enlarged observed boxes can also omit genuine object returns or retain contaminants. These are limitations requiring targeted probes before changing the model; this review does not establish either as the cause of a particular track failure.

Occluders currently use tracked box-center depths rather than mask-associated median return depths. `valid = ~hidden | own_mask` intentionally keeps every target-positive pixel supervised and ignores hidden background pixels; that expression alone is not a demonstrated bug. Incorrect mask overlaps or incorrect depth ordering remain hypotheses. Ground is fixed during this fit rather than recomputed after fitting, so exact exported placement matters.

No acceptance thresholds should be changed merely to improve results on one bag. The older guidance's 10–20 PCA components, full-SE(3) pose, Chamfer description, and assertion that per-point lifting is SLF's segmentation stage should be read as prior project planning, not faithful paper definitions.
