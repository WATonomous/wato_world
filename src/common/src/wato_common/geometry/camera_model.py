"""Calibrated forward-hemisphere camera projection and inverse pixel rays.

The image calibration is used as supplied: full K, followed/preceded by the
declared ROS lens model. Rectification belongs to ingest, not these equations.
Supported distortion models are plumb_bob (4/5 coefficients), rational_polynomial
(8), and equidistant/fisheye (4). Unknown models fail rather than becoming an
undistorted camera. Inverse distortion is checked for convergence.
"""

from dataclasses import dataclass, field

import numpy as np


def validate_transform(transform: np.ndarray) -> np.ndarray:
    """A finite, proper rigid transform; reject scale, reflection and bad rows."""
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("camera transform must be a finite 4x4 matrix")
    r = t[:3, :3]
    if (
        not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)
    ):
        raise ValueError("camera transform must be SE(3), without scale or reflection")
    return t


@dataclass(frozen=True)
class CameraModel:
    """Projection for the ingested image, not the raw CameraInfo metadata.

    Points/rays use optical coordinates (right, down, forward). Depth means
    optical z, not Euclidean range. Fisheye pixels outside the forward
    hemisphere are explicitly unsupported by this pipeline.
    """

    K: np.ndarray
    distortion: tuple[float, ...] = ()
    distortion_model: str = ""
    _inverse_K: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        k = np.array(self.K, dtype=np.float64, copy=True)
        if (
            k.shape != (3, 3)
            or not np.isfinite(k).all()
            or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0)
            or k[0, 0] <= 0
            or k[1, 1] <= 0
        ):
            raise ValueError(
                "K must be finite with positive focal lengths and last row [0, 0, 1]"
            )
        try:
            inverse = np.linalg.inv(k)
        except np.linalg.LinAlgError as exc:
            raise ValueError("K must be invertible") from exc
        d = tuple(float(v) for v in self.distortion)
        if not np.isfinite(d).all():
            raise ValueError("camera distortion coefficients must be finite")
        model = self.distortion_model
        sizes = {
            "": (0,),
            "none": (0,),
            "plumb_bob": (0, 4, 5),
            "rational_polynomial": (8,),
            "equidistant": (4,),
            "fisheye": (4,),
        }
        if model not in sizes:
            raise ValueError(f"unsupported camera distortion model {model!r}")
        if len(d) not in sizes[model]:
            raise ValueError(
                f"distortion model {model!r} needs {sizes[model]} coefficients, got {len(d)}"
            )
        k.flags.writeable = False
        inverse.flags.writeable = False
        object.__setattr__(self, "K", k)
        object.__setattr__(self, "distortion", d)
        object.__setattr__(self, "_inverse_K", inverse)

    @property
    def fisheye(self) -> bool:
        return self.distortion_model in ("equidistant", "fisheye")

    @property
    def nonlinear(self) -> bool:
        return self.fisheye or any(self.distortion)

    def _brown(self, xy):
        """Brown-Conrady/rational distortion and its analytic 2x2 Jacobian."""
        d = np.zeros(8)
        d[: len(self.distortion)] = self.distortion
        k1, k2, p1, p2, k3, k4, k5, k6 = d
        x, y = xy.T
        r2 = x * x + y * y
        numerator = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        denominator = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        radial = numerator / denominator
        derivative = (
            (k1 + r2 * (2 * k2 + 3 * k3 * r2)) * denominator
            - numerator * (k4 + r2 * (2 * k5 + 3 * k6 * r2))
        ) / denominator**2
        out = np.column_stack(
            [
                x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
            ]
        )
        # A folded radial map has multiple inverse rays. Do not let a
        # positive 2D determinant disguise two negative radial eigenvalues.
        usable = (denominator > 0) & (radial > 0) & (radial + 2 * r2 * derivative > 0)
        out[~usable] = np.nan
        cross = 2 * x * y * derivative + 2 * p1 * x + 2 * p2 * y
        jac = np.array(
            [
                [radial + 2 * x * x * derivative + 2 * p1 * y + 6 * p2 * x, cross],
                [cross, radial + 2 * y * y * derivative + 6 * p1 * y + 2 * p2 * x],
            ]
        ).transpose(2, 0, 1)
        return out, jac

    def _theta(self, theta):
        k1, k2, k3, k4 = self.distortion
        s = theta * theta
        value = theta * (1 + s * (k1 + s * (k2 + s * (k3 + s * k4))))
        derivative = 1 + s * (3 * k1 + s * (5 * k2 + s * (7 * k3 + s * 9 * k4)))
        return value, derivative

    def distort(self, xy: np.ndarray) -> np.ndarray:
        """Normalized pinhole directions → normalized distorted image points."""
        if not self.nonlinear:
            return xy.copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if not self.fisheye:
                out, jac = self._brown(xy)
                determinant = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
                out[determinant <= 0] = np.nan
                return out
            radius = np.hypot(xy[:, 0], xy[:, 1])
            theta_d, derivative = self._theta(np.arctan(radius))
            scale = np.divide(
                theta_d, radius, out=np.ones_like(radius), where=radius > 0
            )
            out = xy * scale[:, None]
            out[derivative <= 0] = np.nan
            return out

    def _undistort(self, target: np.ndarray) -> np.ndarray:
        if not self.nonlinear or len(target) == 0:
            return target.copy()
        # Newton steps with backtracking, evaluated in normalized image space.
        # No silently accepted fixed iteration count: the forward residual is
        # checked before rays can enter the optimizer.
        if self.fisheye:
            radius = np.hypot(target[:, 0], target[:, 1])
            theta = np.minimum(radius, 1.4)
            for _ in range(50):
                value, derivative = self._theta(theta)
                error = value - radius
                active = np.abs(error) > 1e-12
                if not active.any():
                    break
                ids = np.flatnonzero(active)
                if (derivative[ids] <= 0).any():
                    raise ValueError(
                        "non-invertible fisheye distortion in requested pixels"
                    )
                step = error[ids] / derivative[ids]
                scale = np.ones(len(ids))
                accepted = np.zeros(len(ids), bool)
                for _ in range(20):
                    trial = theta[ids] - scale * step
                    trial_value, trial_derivative = self._theta(trial)
                    good = (
                        (trial >= 0)
                        & (trial < np.pi / 2)
                        & (trial_derivative > 0)
                        & (np.abs(trial_value - radius[ids]) < np.abs(error[ids]))
                    )
                    use = good & ~accepted
                    theta[ids[use]] = trial[use]
                    accepted |= good
                    if accepted.all():
                        break
                    scale[~accepted] *= 0.5
                if not accepted.all():
                    raise ValueError(
                        "fisheye pixel is outside the invertible forward hemisphere"
                    )
            residual = self._theta(theta)[0] - radius
            if not np.isfinite(theta).all() or (np.abs(residual) > 1e-10).any():
                raise ValueError("fisheye inverse distortion did not converge")
            scale = np.divide(
                np.tan(theta), radius, out=np.ones_like(radius), where=radius > 0
            )
            return target * scale[:, None]

        xy = target.copy()
        for _ in range(50):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                value, jac = self._brown(xy)
            error = value - target
            active = np.max(np.abs(error), axis=1) > 1e-12
            if not active.any():
                break
            ids = np.flatnonzero(active)
            j = jac[ids]
            det = j[:, 0, 0] * j[:, 1, 1] - j[:, 0, 1] * j[:, 1, 0]
            if not np.isfinite(j).all() or (det <= 1e-14).any():
                raise ValueError("non-invertible lens distortion in requested pixels")
            e = error[ids]
            step = np.column_stack(
                [
                    (j[:, 1, 1] * e[:, 0] - j[:, 0, 1] * e[:, 1]) / det,
                    (-j[:, 1, 0] * e[:, 0] + j[:, 0, 0] * e[:, 1]) / det,
                ]
            )
            old_norm = np.linalg.norm(e, axis=1)
            scale = np.ones(len(ids))
            accepted = np.zeros(len(ids), bool)
            start = xy[ids].copy()
            for _ in range(20):
                trial = start - scale[:, None] * step
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    projected = self.distort(trial)
                good = np.linalg.norm(projected - target[ids], axis=1) < old_norm
                use = good & ~accepted
                xy[ids[use]] = trial[use]
                accepted |= good
                if accepted.all():
                    break
                scale[~accepted] *= 0.5
            if not accepted.all():
                raise ValueError("lens inverse distortion did not converge")
        residual = self.distort(xy) - target
        if (
            not np.isfinite(xy).all()
            or not np.isfinite(residual).all()
            or (np.abs(residual) > 1e-10).any()
        ):
            raise ValueError("lens inverse distortion did not converge")
        return xy

    def unproject(self, pixels: np.ndarray, depth_m=1.0) -> np.ndarray:
        """Nx2 image pixels + optical z depths → Nx3 optical camera points."""
        uv = np.asarray(pixels, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
            raise ValueError("pixels must be a finite Nx2 array")
        depth = np.broadcast_to(np.asarray(depth_m, dtype=np.float64), (len(uv),))
        if not np.isfinite(depth).all() or (depth <= 0).any():
            raise ValueError("optical depths must be finite and positive")
        homogeneous = np.column_stack([uv, np.ones(len(uv))]) @ self._inverse_K.T
        xy = self._undistort(homogeneous[:, :2] / homogeneous[:, 2:])
        return np.column_stack([xy, np.ones(len(uv))]) * depth[:, None]

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Nx2 image pixels → Nx3 unit optical camera rays."""
        rays = self.unproject(pixels)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def project(self, points: np.ndarray, cam_T_world: np.ndarray, image_size=None):
        """World points → (pixels, optical z, valid), with one camera model.

        Nonfinite, behind-camera or nonprojectable points have NaN pixels,
        zero depth and False validity. Off-image finite points keep their
        pixels/depth but are invalid when image_size is supplied.
        """
        xyz = np.asarray(points, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("points must be an Nx3 array")
        transform = validate_transform(cam_T_world)
        if image_size is not None:
            if len(image_size) != 2 or any(
                not np.isfinite(v) or v <= 0 for v in image_size
            ):
                raise ValueError("image_size must be positive (width, height)")
        with np.errstate(over="ignore", invalid="ignore"):
            camera = xyz @ transform[:3, :3].T + transform[:3, 3]
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
        uv = np.full((len(xyz), 2), np.nan)
        depth = np.zeros(len(xyz))
        ids = np.flatnonzero(valid)
        if len(ids):
            xy = self.distort(camera[ids, :2] / camera[ids, 2:])
            with np.errstate(over="ignore", invalid="ignore"):
                image = np.column_stack([xy, np.ones(len(ids))]) @ self.K.T
                image = image[:, :2] / image[:, 2:]
            good = np.isfinite(image).all(axis=1)
            valid[ids[~good]] = False
            ids = ids[good]
            uv[ids] = image[good]
            depth[ids] = camera[ids, 2]
        if image_size is not None:
            width, height = image_size
            valid &= (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
        return uv, depth, valid
