import torch
from einops import rearrange


def create_image_grid(shape, device=None):
    """
    Create a normalized coordinate grid for a 2D image.

    Args:
        height (int): Height of the image
        width (int): Width of the image
        device (torch.device): Target device (e.g., 'cuda' or 'cpu')

    Returns:
        coords (Tensor): [H*W, 2] flattened grid of (y, x) coordinates in [-1, 1]
    """
    ys = torch.linspace(-1.0, 1.0, shape[0], device=device)
    xs = torch.linspace(-1.0, 1.0, shape[1], device=device)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)  # [H, W, 2]
    return rearrange(grid, "h w c -> (h w) c")  # preserves row-major flattening


def create_video_grid(temporal, height, width, device=None):
    """
    Create a normalized coordinate grid for 3D video data.

    Args:
        temporal (int): Number of frames (T)
        height (int): Height of each frame (H)
        width (int): Width of each frame (W)
        device (torch.device or str, optional): Device for the tensor

    Returns:
        coords (Tensor): [T, H*W, 3] normalized grid coordinates (t, y, x) ∈ [-1, 1]
    """
    ts = torch.linspace(-1.0, 1.0, temporal, device=device)
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)

    # Generate full 3D grid: [T, H, W, 3]
    grid_t, grid_y, grid_x = torch.meshgrid(ts, ys, xs, indexing="ij")
    grid = torch.stack([grid_t, grid_y, grid_x], dim=-1)  # [T, H, W, 3]

    # Rearrange to [T, H*W, 3] (per-frame flat grid)
    return rearrange(grid, "t h w c -> t (h w) c")


def partition_coords_by_region(coords, order, num_regions, H, W, device=None):
    """
    Pixel-aligned partitioning of a flattened grid into spatial regions.
    Returns:
      - region_to_indices: dict[int -> 1D LongTensor] (pixel ids per region)
      - full_region_ids:  1D LongTensor of shape [H*W] with region id per pixel
      - region_h, region_w: nominal region sizes (may vary by at most 1 if not divisible)
    """
    device = coords.device if device is None else device
    N = coords.shape[0]
    assert N == H * W, f"Grid size mismatch: {N} != {H}*{W}"

    region_to_indices = {}
    full_region_ids = torch.full((H * W,), -1, dtype=torch.long, device=device)

    if order == "rowwise":
        # Split rows into num_regions parts (remainder distributed)
        boundaries = (
            torch.linspace(0, H, steps=num_regions + 1, device=device).round().long()
        )
        for r in range(num_regions):
            r0, r1 = boundaries[r].item(), boundaries[r + 1].item()
            rows = torch.arange(r0, r1, device=device)
            cols = torch.arange(W, device=device)
            # pixel ids: i*W + j
            idx = (rows[:, None] * W + cols[None, :]).reshape(-1)
            region_to_indices[r] = idx
            full_region_ids[idx] = r
        region_h = torch.diff(boundaries).float().mean().item()
        region_w = W

    elif order == "colwise":
        boundaries = (
            torch.linspace(0, W, steps=num_regions + 1, device=device).round().long()
        )
        rows = torch.arange(H, device=device)
        for c in range(num_regions):
            c0, c1 = boundaries[c].item(), boundaries[c + 1].item()
            cols = torch.arange(c0, c1, device=device)
            idx = (rows[:, None] * W + cols[None, :]).reshape(-1)
            region_to_indices[c] = idx
            full_region_ids[idx] = c
        region_h = H
        region_w = torch.diff(boundaries).float().mean().item()

    elif order == "raster":
        # n x n grid; handle non-divisible H/W by balanced boundaries
        n = int(num_regions**0.5)
        assert n * n == num_regions, "num_regions must be a perfect square for 'raster'"
        yb = torch.linspace(0, H, steps=n + 1, device=device).round().long()
        xb = torch.linspace(0, W, steps=n + 1, device=device).round().long()

        rid = 0
        for r in range(n):
            r0, r1 = yb[r].item(), yb[r + 1].item()
            rows = torch.arange(r0, r1, device=device)
            for c in range(n):
                c0, c1 = xb[c].item(), xb[c + 1].item()
                cols = torch.arange(c0, c1, device=device)
                idx = (rows[:, None] * W + cols[None, :]).reshape(-1)
                region_to_indices[rid] = idx
                full_region_ids[idx] = rid
                rid += 1
        region_h = torch.diff(yb).float().mean().item()
        region_w = torch.diff(xb).float().mean().item()

    else:
        raise NotImplementedError(f"Unknown order: {order}")

    return region_to_indices, full_region_ids, region_h, region_w


def gather_sampled_pixels(images, sampled_indices):
    """
    Gathers region-specific pixels from a batch of images using shared flat indices.

    Args:
        images (Tensor): [B, C, H, W] input image batch.
        sampled_idices (Tensor): [N] 1D flattened indices (H*W) shared across the batch selected from the grid.

    Returns:
        Tensor: [B, C, N] gathered pixels.
    """
    B, C, _, _ = images.shape
    N = sampled_indices.size(0)

    # Flatten spatial dimensions
    images_flat = images.view(B, C, -1)  # [B, C, H*W]

    # Expand indices to all batches
    expanded_idx = sampled_indices.unsqueeze(0).expand(B, -1)  # [B, N]

    # Gather selected pixels
    gathered = torch.gather(
        images_flat,
        dim=2,
        index=expanded_idx.unsqueeze(1).expand(-1, C, -1),  # [B, C, N]
    )

    return gathered  # [B, C, N]


def assign_coords_to_regions(coords, order: str, num_regions: int):
    x, y = coords[:, 0], coords[:, 1]

    if order == "colwise":
        step = 2.0 / num_regions
        region_ids = ((x + 1.0) / step).floor().clamp(0, num_regions - 1)

    elif order == "rowwise":
        step = 2.0 / num_regions
        region_ids = ((y + 1.0) / step).floor().clamp(0, num_regions - 1)

    elif order == "raster":
        sqrt_K = int(num_regions**0.5)
        assert sqrt_K * sqrt_K == num_regions, "num_regions must be a square"
        step = 2.0 / sqrt_K
        ix = ((x + 1.0) / step).floor().clamp(0, sqrt_K - 1)
        iy = ((y + 1.0) / step).floor().clamp(0, sqrt_K - 1)
        region_ids = iy * sqrt_K + ix

    else:
        raise ValueError(f"Unknown order: {order}")

    return region_ids.long()
