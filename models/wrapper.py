from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from common.modularization_utils import (
    create_image_grid,
    create_video_grid,
    partition_coords_by_region,
)
from .fim import FIMWeighter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exists(val):
    return val is not None


class MetaWrapper(nn.Module):
    def __init__(self, P, decoder):
        super().__init__()
        self.P = P
        self.decoder = decoder
        self.data_type = P.data_type
        self.use_fim = P.fim
        self.fim_weighter = (
            FIMWeighter(
                self.decoder,
                lambda_fim=P.fim_lambda,
                beta_ema=P.fim_beta,
                epsilon=P.fim_epsilon,
            )
            if self.use_fim
            else None
        )

        if self.data_type == "img":
            self.full_height = P.data_size[1]
            self.full_width = P.data_size[2]
            mgrid = create_image_grid(
                (self.full_height, self.full_width), device=device
            )
            (
                self.region_to_indices,
                full_region_ids,
                self.patch_height,
                self.patch_width,
            ) = partition_coords_by_region(
                coords=mgrid,
                order=P.order,
                num_regions=P.num_submodules,
                H=self.full_height,
                W=self.full_width,
                device=device,
            )
            self.register_buffer("full_region_ids", full_region_ids)

        elif self.data_type == "video":  # Not updated
            self.temporal = P.data_size[0]
            self.full_h = P.data_size[2]
            self.full_w = P.data_size[3]
            mgrid = create_video_grid(
                self.temporal, self.full_h, self.full_w, device=device
            )
        elif self.data_type == "ray":
            mgrid = None  # no predefined grid
        else:
            raise NotImplementedError()

        if mgrid is not None:
            self.register_buffer("grid", mgrid)

    # def get_batch_params(self, params, batch_size):
    #     if params is None:
    #         params = OrderedDict()
    #         for name, param in self.decoder.meta_named_parameters():
    #             params[name] = param[None, ...].repeat(
    #                 (batch_size,) + (1,) * len(param.shape)
    #             )
    #     return params

    def get_batch_params(self, params, batch_size):
        if params is None:
            params = OrderedDict()
            for name, p in self.decoder.meta_named_parameters():
                b = p.unsqueeze(0).expand(batch_size, *p.shape).clone().requires_grad_()
                params[name] = b
        return params

    def coord_init(self):
        self.support_coord = None
        self.support_indices = None
        self.query_coord = None
        self.query_indices = None

    def get_batch_coords(self, inputs=None, params=None, step_inner=-1):
        if inputs is None and params is None:
            meta_batch_size = 1
        elif inputs is None:
            meta_batch_size = list(params.values())[0].size(0)
        else:
            meta_batch_size = inputs.size(0)

        if self.support_coord is None:
            coords = getattr(self, "grid", None)
            indices = torch.arange(coords.shape[0], device=coords.device)
        else:
            coords, indices = self.get_patch_coords(self.support, step_inner)

        if self.data_type == "video" and coords is not None and coords.dim() == 3:
            video_coords = []
            for i in range(coords.size(0)):
                for j in range(coords.size(1)):
                    video_coords.append(
                        torch.cat(
                            [
                                torch.tensor([i], device=coords.device).float(),
                                coords[i][j],
                            ]
                        )
                    )
            coords = torch.stack(video_coords)

        if coords is not None:
            # ? Caustion. Replaced repeat with expand.
            coords = coords.clone().detach()[None, ...].expand(meta_batch_size, -1, -1)
        return coords, indices, meta_batch_size

    def forward(self, inputs, params=None, step_inner=-1, step_iter=-1):
        if self.data_type == "img":
            return self.forward_image(inputs, params, step_inner, step_iter)
        elif self.data_type == "video":
            return self.forward_video(inputs, params, step_inner, step_iter)
        elif self.data_type == "ray":
            return self.forward_ray(inputs, params, step_inner, step_iter)
        else:
            raise NotImplementedError()

    def get_patch_coords(self, support, step_inner=-1):
        """
        Returns the coordinates and indices of the grid points belonging to the specified region
        and phase (support/query) in the current step.

        Args:
            support (bool): Whether to return support (True) or query (False) coordinates
            step_inner (int): The region/submodule ID (0 to num_submodules-1); if -1, uses full grid indices of the set

        Returns:
            coords_in_patch (Tensor): (N_patch, 2) coordinates
            indices_in_patch (Tensor): (N_patch,) indices in the full grid
        """
        if step_inner == -1:
            indices = self.support_indices if support else self.query_indices
        else:
            region_mask = torch.zeros(
                len(self.grid), dtype=torch.bool, device=self.grid.device
            )
            region_mask[self.region_to_indices[step_inner]] = True

            phase_indices = self.support_indices if support else self.query_indices
            phase_mask = torch.zeros(
                len(self.grid), dtype=torch.bool, device=self.grid.device
            )
            phase_mask[phase_indices] = True

            final_mask = region_mask & phase_mask
            indices = final_mask.nonzero(as_tuple=True)[0]

        coords_in_patch = self.grid[indices]
        return coords_in_patch, indices.to(device)

    def scatter_flat_to_image(self, flat_vals, indices, B, C, H, W):
        """
        Projects sparse per-pixel values into a dense image.

        Args:
            flat_vals: (B, N, C)
            indices:   (N,) – flat pixel indices
            B, C, H, W: output dimensions

        Returns:
            (B, C, H, W) dense image with zeros at unselected pixels
        """
        dense = torch.zeros(
            (B, H * W, C), device=flat_vals.device, dtype=flat_vals.dtype
        )
        dense[:, indices, :] = flat_vals
        return rearrange(dense, "b (h w) c -> b c h w", h=H, w=W).contiguous()

    def forward_image(
        self, inputs=None, params=None, step_inner: int = -1, step_iter: int = -1
    ):
        """
        Forward pass for image data in modular or non-modular Neural Field models.

        Supports:
        - Full-image or sampled-pixel evaluation
        - Modular decoders with per-region submodules via `region_ids`
        - Loss computation when ground truth is provided
        - Pure prediction when inputs is None

        Returns:
        if inputs is not None:
            pixel_loss: (B, C, H, W)
            out_img:    (B, C, H, W)
        else:
            out_img:    (B, C, H, W)
        """
        # Episode start!
        is_train = self.training and (inputs is not None)

        if self.use_fim and self.support and step_iter == 0:
            self.fim_weighter.reset_fisher(self.fim_weighter.epsilon)

        b, c, h, w = (
            inputs.shape
            if inputs is not None
            else (1, 3, self.full_height, self.full_width)
        )

        coords, selected_idx, _ = self.get_batch_coords(
            inputs, params, step_inner
        )  # coords: (B, N, 2), selected_idx: (N,)

        # Build per-pixel region ids, aligned to selected_idx
        if is_train and self.support:
            # support step: a single region is active (step_inner)
            region_ids = torch.full(
                selected_idx.shape,
                int(step_inner),
                dtype=torch.long,
                device=coords.device,
            )
        else:
            # query or eval: gather region per pixel in O(1)
            region_ids = self.full_region_ids.index_select(0, selected_idx)

        # Decode (decoder will scatter by region internally and preserve original pixel order)
        out = self.decoder(coords, params=params, region_ids=region_ids)  # (B, N, C)

        # ---- prediction only (no GT) ----
        if inputs is None:
            out_img = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
            return out_img

        # ---- training/eval with ground truth ----
        gt_flat = rearrange(inputs, "b c h w -> b (h w) c")[:, selected_idx, :]
        mse_flat = F.mse_loss(out, gt_flat, reduction="none")  # (B, N, C)
        if self.use_fim:
            fim_weights = self.fim_weighter.compute_weights(
                per_sample_loss_tensor=mse_flat, reduce_dims=2, update_fisher=True
            )

        # For visualization/logging, scatter loss back to full (B, C, H, W)
        pixel_loss = self.scatter_flat_to_image(
            mse_flat, selected_idx, B=b, C=c, H=h, W=w
        )  # (B, C, H, W)

        out_img = (
            rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
            if out.shape[1] == h * w
            else self.scatter_flat_to_image(out, selected_idx, B=b, C=c, H=h, W=w)
        )
        return pixel_loss, out_img

    def forward_video(self, inputs=None, params=None, step_inner=-1, step_iter=-1):
        b, t, c, h, w = inputs.size()
        coords, meta_batch_size = self.get_batch_coords(inputs, params)

        if (
            hasattr(self.decoder, "forward")
            and "routing_idx" in self.decoder.forward.__code__.co_varnames
        ):
            routing_idx = torch.full(
                (meta_batch_size,), step_inner, dtype=torch.long, device=coords.device
            )
            out = self.decoder(
                coords,
                params=params,
                routing_idx=routing_idx,
            )
        else:
            out = self.decoder(coords, params=params)

        out = rearrange(out, "b (t h w) c -> b t c h w", t=t, h=h, w=w)

        if exists(inputs):
            if self.sampled_coord is None:
                return F.mse_loss(inputs, out, reduction="none"), out
            else:
                inputs = rearrange(inputs, "b c h w -> b c (h w)")[
                    :, :, self.sampled_index
                ]
                return (
                    F.mse_loss(
                        inputs.view(meta_batch_size, -1),
                        out.view(meta_batch_size, -1),
                        reduction="none",
                    ).mean(dim=1),
                    out,
                )

        out = rearrange(
            out, "b c (h w) -> b c h w", h=self.patch_height, w=self.patch_width
        )
        0
        return out

    def sample(self, sample_type):
        if sample_type == "random":
            self.random_sample()
        else:
            raise NotImplementedError()

    def random_sample(self):
        coord_size = self.grid.size(0)
        perm = torch.randperm(coord_size)
        ns = int(self.P.data_ratio * coord_size)
        self.support_indices = perm[:ns]
        self.support_coord = self.grid[self.support_indices]

        # Compute complement (query)
        self.query_indices = perm[ns:]  # (Nq,)
        self.query_coord = self.grid[self.query_indices]  # (Nq, 2)
