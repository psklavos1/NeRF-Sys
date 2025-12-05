from typing import Optional, Tuple

import torch
from torch import Tensor

from nerfs.scene_box import SceneBox


# ----------------------- utils -----------------------
def _rays_cam_to_world(
    dirs_cam: Tensor,  # (..., 3)
    c2w: Tensor,  # (3,4) or (4,4)
) -> Tuple[Tensor, Tensor]:
    """
    Convert camera-frame directions to world-frame origins & directions.
    - dirs_cam: (..., 3)
    - c2w: top-left 3x3 used for rotation, [:3,3] for translation.
    Returns origins_world, dirs_world with leading dims = dirs_cam.shape[:-1].
    """
    orig_shape = dirs_cam.shape
    dirs_flat = dirs_cam.reshape(-1, 3)

    R = c2w[:3, :3]  # (3,3)
    t = c2w[:3, 3]  # (3,)

    dirs_w = dirs_flat @ R.T
    org_w = t.expand_as(dirs_w)

    return org_w.reshape(*orig_shape), dirs_w.reshape(*orig_shape)


# -------------------------------- Packing API -------------
def pack_rays(
    rays_o: Tensor,  # (H,W,3)
    rays_d: Tensor,  # (H,W,3)
    near: Tensor,  # (H,W,1)
    far: Tensor,  # (H,W,1)
) -> Tensor:
    """Return (H, W, 8) = [o(3), d(3), near, far]."""
    return torch.cat([rays_o, rays_d, near, far], dim=-1)


def unpack_rays(rays: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """rays: (N, 8) or (H, W, 8) -> (origins, dirs, near, far) with shapes:
    origins=(N,3), dirs=(N,3), near=(N,1), far=(N,1)
    """
    assert rays.shape[-1] == 8, "packed rays must be (..., 8)"
    flat = rays.view(-1, 8).contiguous()
    origins = flat[:, :3]
    dirs = flat[:, 3:6]
    near = flat[:, 6:7]
    far = flat[:, 7:8]
    return origins, dirs, near, far


# ----------------------- Public API -----------------------
def get_rays(
    directions: torch.Tensor,  # (H, W, 3) or (N, 3) in camera coords (unit)
    c2w: torch.Tensor,  # (3,4) or (4,4)
    scene_box: Optional["SceneBox"] = None,
    near: Optional[float] = None,  # used only when scene_box is None
    far: Optional[float] = None,  # used only when scene_box is None
    *,
    aabb_max_bound: float = 1e10,
    aabb_invalid_value: float = 1e10,
) -> torch.Tensor:
    """
    Returns per-ray tensors as either:
      - (H, W, 8) if input was (H, W, 3)
      - (N, 8)    if input was (N, 3)

    Each ray = [ox, oy, oz, dx, dy, dz, near, far].

    If `scene_box` is provided:
      - near/far are computed from ray–AABB intersection.
    Else:
      - `near` and `far` must be floats (broadcasted).
    """

    # --------------------------------------------------------
    # 1. Normalize input shape
    # --------------------------------------------------------
    if directions.ndim == 2 and directions.shape[1] == 3:
        flat_input = True
        H = W = None
    elif directions.ndim == 3 and directions.shape[-1] == 3:
        flat_input = False
        H, W, _ = directions.shape
    else:
        raise ValueError(
            f"directions must be (H, W, 3) or (N, 3), got {tuple(directions.shape)}"
        )

    # --------------------------------------------------------
    # 2. Camera → world transform (reuse helper)
    # --------------------------------------------------------
    rays_o, rays_d = _rays_cam_to_world(directions, c2w)
    # shapes: (H,W,3) or (N,3)

    # Flatten for intersection or broadcasting
    rays_o_flat = rays_o.reshape(-1, 3)
    rays_d_flat = rays_d.reshape(-1, 3)

    # --------------------------------------------------------
    # 3. Compute near/far
    # --------------------------------------------------------
    if scene_box is not None:
        # Ray-box intersection (returns per-ray tmin/tmax)
        tmin, tmax = scene_box.ray_aabb_intersect(
            rays_o_flat,
            rays_d_flat,
            eps=1e-8,
            max_bound=aabb_max_bound,
            invalid_value=aabb_invalid_value,
        )
        near_vals = tmin.unsqueeze(-1)  # (N,1)
        far_vals = tmax.unsqueeze(-1)  # (N,1)
    else:
        assert (
            near is not None and far is not None
        ), "Provide near/far floats when scene_box is None"
        N = rays_o_flat.shape[0]
        near_vals = torch.full(
            (N, 1), float(near), dtype=rays_o_flat.dtype, device=rays_o_flat.device
        )
        far_vals = torch.full(
            (N, 1), float(far), dtype=rays_o_flat.dtype, device=rays_o_flat.device
        )

    # --------------------------------------------------------
    # 4. Pack and reshape back
    # --------------------------------------------------------
    if flat_input:
        return torch.cat(
            [rays_o_flat, rays_d_flat, near_vals, far_vals], dim=-1
        )  # (N, 8)
    else:
        near_img = near_vals.view(H, W, 1)
        far_img = far_vals.view(H, W, 1)
        return pack_rays(rays_o, rays_d, near_img, far_img)  # (H, W, 8)


def get_ray_directions(
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    center_pixels: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns camera-frame directions (H, W, 3), unit-normalized.
    Suitable for RUB(Rotation of camera) c2w transormation.
    Pinhole (pixel center at (cx,cy)):
      x_cam =  (i - cx) / fx
      y_cam = -(j - cy) / fy   # image j increases downward -> y_cam up
      z_cam = -1               # camera looks along -Z
    """
    # j: rows [0..H-1], i: cols [0..W-1]
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    if center_pixels:
        i = i + 0.5
        j = j + 0.5

    dirs = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)],
        dim=-1,  # (H, W, 3)
    )
    # unit-normalize (protect against 0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min_(1e-12)
    return dirs


@torch.no_grad()
def clamp_rays_near_far(
    rays: torch.Tensor,  # (N, 8) = [ox,oy,oz, dx,dy,dz, near, far]
    near_far_override: tuple[float | None, float | None] | None,
    *,
    eps: float = 1e-6,
    invalid_value: float = float("inf"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Clamp packed rays' near/far with optional overrides.
    Returns (rays_clamped, valid_mask). Invalid rays get both near/far = invalid_value.
    """
    if near_far_override is None:
        # validity as "finite and far>near"
        near = rays[:, 6]
        far = rays[:, 7]
        valid = torch.isfinite(near) & torch.isfinite(far) & (far > near + eps)
        return rays, valid

    n_override, f_override = near_far_override
    rays = rays.clone()
    near = rays[:, 6]
    far = rays[:, 7]

    if n_override is not None:
        n_val = torch.as_tensor(float(n_override), device=rays.device, dtype=rays.dtype)
        near = torch.maximum(near, n_val)
    if f_override is not None:
        f_val = torch.as_tensor(float(f_override), device=rays.device, dtype=rays.dtype)
        far = torch.minimum(far, f_val)

    valid = torch.isfinite(near) & torch.isfinite(far) & (far > near + eps)
    inv = torch.full_like(near, float(invalid_value))
    near = torch.where(valid, near, inv)
    far = torch.where(valid, far, inv)

    rays[:, 6] = near
    rays[:, 7] = far
    return rays, valid
