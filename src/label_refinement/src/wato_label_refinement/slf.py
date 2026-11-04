"""SLF-inspired rounded-box refinement (Li et al., ECCV 2024) per track.

One shape per track, fit to the track's masks in every camera view of it and
to its LiDAR points:

- **Shape**: a rounded box (signed distance function) with one length, width
  and height for the whole track.
- **Pose**: Phase 3's (refine.py). Starting from its smoothed positions,
  each frame's box follows size changes at the observed near corner
  (refine.Anchor), so the size
  decides where the far faces are: what the LiDAR doesn't see and the
  silhouette does. A parked vehicle's one box keeps its centre. One small
  offset along the box's axes is shared by the track; the heading is Phase
  3's; the bottom is Phase 3's ground.
- **Losses**: Dice between each view's soft silhouette (the SDF's minimum
  along each pixel's ray) and its mask, over the pixels no nearer object's
  mask covers (SLF's occlusion map); a robust |SDF| at the LiDAR points, which
  lie on the surface (for a pedestrian, only the SDF of points outside: its box
  encloses them); weak priors on the size (the class's mean) and on the
  offset.
- **Optimiser**: Adam over log size and the offset (SLF's lr 0.1, 150 steps),
  many tracks at once (fit_many), each on its own loss.

Torch only here: views.py builds a Problem with the shared camera model.
Optimization uses track-local coordinates; Fit returns exactly the evaluated
world positions for every output frame. This is a constrained adaptation,
not the paper's latent-shape and per-frame pose optimizer. See
docs/research/slf_projection_implementation_review.md for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from wato_label_refinement.config import SlfConfig

# Smooths the SDF's corners and |SDF| at zero, metres.
_EPS_M = 1e-3


@dataclass(frozen=True)
class Problem:
    """One track's fit, as arrays (F frames, V views of P pixels, N points)."""

    size0: np.ndarray  # (3,) l, w, h to start from (Phase 3's)
    prior: Optional[np.ndarray]  # (3,) the class's mean l, w, h; None: no prior
    # Per frame: the box's heading and where its centre is for a size.
    heading: np.ndarray  # (F,)
    centre_obs: np.ndarray  # (F, 2) reference origin reproducing Phase 3 at size0
    anchor_lo: np.ndarray  # (F, 2) refine.Anchor lo along the length, width axes
    anchor_hi: np.ndarray  # (F, 2)
    anchor_near_lo: np.ndarray  # (F, 2) bool
    anchored: np.ndarray  # (F,) bool; False: the centre doesn't follow the size
    ground_z: np.ndarray  # (F,) the box's bottom
    # Per view.
    view_frame: np.ndarray  # (V,) int, its frame
    view_shift: np.ndarray  # (V, 2) the object's motion, frame time → image time
    origin: np.ndarray  # (V, 3) the camera's centre, world
    dirs: np.ndarray  # (V, P, 3) unit ray per crop pixel, world
    target: np.ndarray  # (V, P) 1 = the track's mask
    valid: np.ndarray  # (V, P) bool: not a nearer object's mask
    footprint: np.ndarray  # (V,) metres per crop pixel at the box's depth
    # LiDAR returns on the object.
    points: np.ndarray  # (N, 3) world
    point_frame: np.ndarray  # (N,) int
    # True: the returns need only lie inside the box (a person's box encloses
    # arms and legs, its faces aren't on the returns); False: on its surface.
    enclose: bool = False


@dataclass(frozen=True)
class Fit:
    size: np.ndarray  # (3,) l, w, h
    offset: np.ndarray  # (2,) along the length, width axes
    centres: np.ndarray  # (F, 3) exact evaluated output positions, every frame
    # Mean over views of 1 − IoU(silhouette, mask), and the median |SDF| at
    # the points (m), at the start and after the fit.
    residual_silhouette: float
    residual_lidar: float
    start_residual_silhouette: float
    start_residual_lidar: float


def device(name: str) -> "torch.device":  # noqa: F821
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rounded_box_sdf(p, half, radius):
    """Signed distance from local points p (..., 3) to a box of half extents
    `half` (..., 3) with edges rounded by `radius` (broadcast like p[..., 0])."""
    import torch

    r = torch.as_tensor(radius)
    q = p.abs() - (half - r[..., None])
    outside = torch.sqrt(q.clamp(min=0).square().sum(-1) + _EPS_M**2) - _EPS_M
    return outside + q.amax(-1).clamp(max=0) - r


class _Batch:
    """Several tracks' fits as one: every frame, view and point carries its
    track's index, and each track's loss is its own (they are summed, so the
    tracks don't interact and Adam, elementwise, steps each as if alone)."""

    def __init__(self, probs: list[Problem], cfg: SlfConfig, dev):
        import torch

        def t(xs, dtype=torch.float32):
            return torch.as_tensor(np.concatenate(xs), dtype=dtype, device=dev)

        def each(xs):
            return torch.as_tensor(
                np.concatenate([np.full(len(x), b) for b, x in enumerate(xs)]),
                dtype=torch.long,
                device=dev,
            )

        self.cfg, self.n = cfg, len(probs)
        # World maps can use large coordinates (e.g. projected map grids).
        # Subtract each track's origin in float64 before making float32
        # optimization tensors; otherwise centimetres disappear on input.
        origins = [np.r_[p.centre_obs[0], p.ground_z[0]] for p in probs]
        self.frame_origins = torch.as_tensor(
            np.concatenate(
                [
                    np.repeat(o[None], len(p.heading), axis=0)
                    for p, o in zip(probs, origins)
                ]
            ),
            dtype=torch.float64,
            device=dev,
        )
        f0 = np.cumsum([0] + [len(p.heading) for p in probs])[:-1]
        self.frame_track = each([p.heading for p in probs])
        self.frame_ranges = np.cumsum([0] + [len(p.heading) for p in probs])
        self.cos = t([np.cos(p.heading) for p in probs])
        self.sin = t([np.sin(p.heading) for p in probs])
        self.centre_obs = t([p.centre_obs - o[:2] for p, o in zip(probs, origins)])
        self.lo, self.hi = (
            t([p.anchor_lo for p in probs]),
            t([p.anchor_hi for p in probs]),
        )
        self.near_lo = t([p.anchor_near_lo for p in probs], torch.bool)
        self.anchored = t([p.anchored for p in probs], torch.bool)
        self.ground = t([p.ground_z - o[2] for p, o in zip(probs, origins)])
        self.view_track = each([p.view_frame for p in probs])
        self.view_frame = t(
            [p.view_frame + f0[b] for b, p in enumerate(probs)], torch.long
        )
        self.view_shift = t([p.view_shift for p in probs])
        self.origin, self.dirs = (
            t([p.origin - o for p, o in zip(probs, origins)]),
            t([p.dirs for p in probs]),
        )
        self.target = t([p.target for p in probs])
        self.valid = t([p.valid for p in probs], torch.bool)
        self.footprint = t([p.footprint for p in probs])
        self.points = t([p.points - o for p, o in zip(probs, origins)])
        self.point_track = each([p.points for p in probs])
        self.point_frame = t(
            [p.point_frame + f0[b] for b, p in enumerate(probs)], torch.long
        )
        self.enclose = torch.as_tensor([p.enclose for p in probs], device=dev)[
            self.point_track
        ]
        self.view_ranges = np.cumsum([0] + [len(p.view_frame) for p in probs])
        self.point_ranges = np.cumsum([0] + [len(p.points) for p in probs])
        self.n_views = torch.as_tensor(np.diff(self.view_ranges), device=dev).clamp(
            min=1
        )
        self.n_points = torch.as_tensor(np.diff(self.point_ranges), device=dev).clamp(
            min=1
        )
        self.has_prior = torch.as_tensor(
            [p.prior is not None for p in probs], device=dev
        )
        self.prior = torch.log(
            torch.as_tensor(
                np.array([p.prior if p.prior is not None else p.size0 for p in probs]),
                dtype=torch.float32,
                device=dev,
            )
        )
        self.steps = torch.linspace(-1.0, 1.0, cfg.ray_samples, device=dev)

    def _centres(self, size, offset):
        """(F, 3) track-local centres for sizes (B, 3) and offsets (B, 2)."""
        import torch

        s = size[self.frame_track]
        ext = s[:, :2]
        full = (self.hi - self.lo) >= ext
        along = torch.where(
            full,
            (self.lo + self.hi) / 2,
            torch.where(self.near_lo, self.lo + ext / 2, self.hi - ext / 2),
        )
        along = torch.where(self.anchored[:, None], along, torch.zeros_like(along))
        along = along + offset[self.frame_track]
        x = self.centre_obs[:, 0] + along[:, 0] * self.cos - along[:, 1] * self.sin
        y = self.centre_obs[:, 1] + along[:, 0] * self.sin + along[:, 1] * self.cos
        return torch.stack([x, y, self.ground + s[:, 2] / 2], -1)

    def centres(self, size, offset):
        """The evaluated centres restored to float64 world coordinates."""
        return self._centres(size, offset).double() + self.frame_origins

    def _local(self, p, centre, cos, sin):
        import torch

        d = p - centre
        return torch.stack(
            [
                d[..., 0] * cos + d[..., 1] * sin,
                -d[..., 0] * sin + d[..., 1] * cos,
                d[..., 2],
            ],
            -1,
        )

    def radius(self, size):
        """(B,) edge radius per track."""
        return self.cfg.corner_radius_frac * size.min(1).values.detach() / 2

    def silhouettes(self, size, offset):
        """(V, P) soft silhouettes."""
        import torch

        c = self._centres(size, offset)[self.view_frame]
        c = torch.cat([c[:, :2] + self.view_shift, c[:, 2:]], -1)  # (V, 3)
        f = self.view_frame
        sv = size[self.view_track]
        reach = (sv / 2).norm(dim=1).detach() + 0.5
        t_near = ((c - self.origin)[:, None, :] * self.dirs).sum(-1)  # (V, P)
        # Samples behind the camera don't count: clamp them to just ahead.
        t = (t_near[..., None] + self.steps * reach[:, None, None]).clamp(min=0.1)
        p = self.origin[:, None, None, :] + self.dirs[:, :, None, :] * t[..., None]
        local = self._local(
            p,
            c[:, None, None, :],
            self.cos[f][:, None, None],
            self.sin[f][:, None, None],
        )
        r = self.radius(size)[self.view_track][:, None, None]
        d = rounded_box_sdf(local, (sv / 2)[:, None, None, :], r).amin(-1)
        edge = self.cfg.edge_px * self.footprint[:, None]
        return torch.sigmoid(-d / edge)

    def point_sdf(self, size, offset):
        """(N,) the points' SDF; outside only (0 inside) for an enclosing fit."""
        import torch

        c = self._centres(size, offset)[self.point_frame]
        f = self.point_frame
        local = self._local(self.points, c, self.cos[f], self.sin[f])
        d = rounded_box_sdf(
            local, size[self.point_track] / 2, self.radius(size)[self.point_track]
        )
        return torch.where(self.enclose, d.clamp(min=0), d)

    def dice(self, sil):
        v = self.valid.float()
        inter = (sil * self.target * v).sum(1)
        denom = (sil * v).sum(1) + (self.target * v).sum(1)
        return 1 - (2 * inter + 1) / (denom + 1)

    def loss(self, log_size, offset):
        import torch

        size = log_size.exp()
        cfg = self.cfg
        per = torch.zeros(self.n, device=log_size.device)
        if self.dirs.shape[0]:
            d = self.dice(self.silhouettes(size, offset))
            per = (
                per
                + cfg.weight_mask
                * torch.zeros_like(per).index_add(0, self.view_track, d)
                / self.n_views
            )
        if self.points.shape[0]:
            s = self.point_sdf(size, offset)
            delta = cfg.lidar_robust_m
            rho = torch.sqrt(s.square() + delta**2) - delta
            per = (
                per
                + cfg.weight_lidar_per_m
                * torch.zeros_like(per).index_add(0, self.point_track, rho)
                / self.n_points
            )
        prior = ((log_size - self.prior) / cfg.size_prior_sigma).square().sum(1)
        per = per + cfg.weight_size_prior * torch.where(
            self.has_prior, prior, torch.zeros_like(prior)
        )
        per = per + cfg.weight_offset * (offset / cfg.offset_sigma_m).square().sum(1)
        return per.sum()

    def residuals(self, size, offset) -> list[tuple[float, float]]:
        """Per track: (mean 1 − IoU over its views, median |SDF| at its points);
        NaN without views / points."""
        import torch

        with torch.no_grad():
            sil = np.full(self.n, np.nan)
            if self.dirs.shape[0]:
                hard = self.silhouettes(size, offset) > 0.5
                tgt = self.target > 0.5
                inter = (hard & tgt & self.valid).sum(1).float()
                union = ((hard | tgt) & self.valid).sum(1).float()
                per_view = (1 - inter / union.clamp(min=1)).cpu().numpy()
                for b in range(self.n):
                    a, z = self.view_ranges[b], self.view_ranges[b + 1]
                    if z > a:
                        sil[b] = float(per_view[a:z].mean())
            lid = np.full(self.n, np.nan)
            if self.points.shape[0]:
                d = self.point_sdf(size, offset).abs().cpu().numpy()
                for b in range(self.n):
                    a, z = self.point_ranges[b], self.point_ranges[b + 1]
                    if z > a:
                        lid[b] = float(np.median(d[a:z]))
        return list(zip(sil.tolist(), lid.tolist()))


def fit_many(probs: list[Problem], cfg: SlfConfig, dev=None) -> list[Fit]:
    """Fit several tracks' sizes and offsets at once (module docstring), as
    if one by one: the GPU's cost is in launching each step, not its size."""
    import torch

    if not probs:
        return []
    dev = dev or device(cfg.device)
    model = _Batch(probs, cfg, dev)
    log_size = torch.tensor(
        np.log(np.array([np.asarray(p.size0, float) for p in probs])),
        dtype=torch.float32,
        device=dev,
    ).requires_grad_()
    offset = torch.zeros((len(probs), 2), device=dev, requires_grad=True)
    start = model.residuals(log_size.detach().exp(), offset.detach())
    opt = torch.optim.Adam([log_size, offset], lr=cfg.lr)
    lo, hi = np.log(cfg.min_size_m), np.log(cfg.max_size_m)
    for _ in range(cfg.iterations):
        opt.zero_grad()
        model.loss(log_size, offset).backward()
        opt.step()
        with torch.no_grad():
            log_size.clamp_(lo, hi)
    size, off = log_size.detach().exp(), offset.detach()
    end = model.residuals(size, off)
    sizes, offs = size.cpu().numpy().astype(float), off.cpu().numpy().astype(float)
    centres = model.centres(size, off).cpu().numpy().astype(float)
    return [
        Fit(
            size=sizes[b],
            offset=offs[b],
            centres=centres[model.frame_ranges[b] : model.frame_ranges[b + 1]],
            residual_silhouette=end[b][0],
            residual_lidar=end[b][1],
            start_residual_silhouette=start[b][0],
            start_residual_lidar=start[b][1],
        )
        for b in range(len(probs))
    ]


def fit(prob: Problem, cfg: SlfConfig, dev=None) -> Fit:
    """Fit one track's size and offset (module docstring)."""
    return fit_many([prob], cfg, dev)[0]
