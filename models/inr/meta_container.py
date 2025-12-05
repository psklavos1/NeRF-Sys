from typing import List, Optional, OrderedDict, Literal, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.image_metadata import ImageMetadata
from models.metamodule import MetaModule
from models.encodings import SHEncoder, FrequencyEncoder
from models.inr.meta_vanilla import MetaNeRF
from models.inr.meta_ngp import MetaNGP


def build_expert(nerf_variant, **nerf_kwargs):
    if nerf_variant == "instant":
        return MetaNGP(**nerf_kwargs)
    return MetaNeRF(**nerf_kwargs)


class MetaContainer(MetaModule):
    """
    Container:
      - Owns routing & background.
      - Soft- or hard-routes queries to sub-experts.
      - (When used with the occupancy renderer) each expert also owns its own occ grid.

    Notes:
      - Soft routing must blend colors σ-aware: c_mix = sum_k w_k σ_k c_k / sum_k w_k σ_k; σ_mix = sum_k w_k σ_k.
      - Keep joint_training warm-up noops to keep optimizer param-state “touched”.
    """

    def __init__(
        self,
        num_submodules: int,
        centroids: torch.Tensor,
        aabb: torch.Tensor,
        nerf_variant: Literal["instant", "vanilla"] = "instant",
        *,
        # ---- routing options ----
        boundary_margin: float = 1.0,
        cluster_2d: bool = True,
        joint_training: bool = False,
        # ---- background head ----
        use_bg_nerf: bool = True,
        bg_hidden: int = 32,
        bg_encoding: Literal["spherical", "fourier"] = "spherical",
        # ---- occupancy (two-tier) ----
        occ_conf: Optional[Dict] = None,
        # ---- forwarded to each expert (hidden sizes, etc.) ----
        **nerf_kwargs,
    ):
        super().__init__()
        assert num_submodules > 0
        assert centroids.ndim == 2 and centroids.size(0) == num_submodules
        assert boundary_margin >= 1.0

        self.use_occ = occ_conf.get("use_occ", False)
        # World-space AABB (kept for potential global logic; experts carry their own)
        self.register_buffer(
            "scene_aabb_vec",
            torch.cat([aabb[0], aabb[1]], dim=0).float(),
            persistent=True,
        )
        self.register_buffer("centroids", centroids.to(torch.float32), persistent=True)
        self.boundary_margin = float(boundary_margin)
        self.cluster_2d = bool(cluster_2d)
        self.joint_training = bool(joint_training)
        self._coord_idx = (1, 2) if self.cluster_2d else (0, 1, 2)

        self.nerf_variant = nerf_variant
        self.dim_out = 4


        # ------------------------------- Experts -------------------------------
        expert_box_list = nerf_kwargs.pop("expert_box_list")

        expert_kwargs_base = {
            **nerf_kwargs,
            "occ_conf": occ_conf,
        }

        self.submodules = nn.ModuleList()
        for i, box in enumerate(expert_box_list):
            # Each `box` is a SceneBox (min/max, etc.) the expert will use.
            kw = {
                **expert_kwargs_base,
                "scene_box": box,
            }
            self.submodules.append(build_expert(nerf_variant, **kw))

        # ------------------------------- Background -------------------------------
        self.use_bg_nerf = bool(use_bg_nerf)
        if self.use_bg_nerf:
            if bg_encoding == "spherical":
                self.bg_dir_enc = SHEncoder(levels=4, implementation="tcnn")
                in_ch_dir = self.bg_dir_enc.out_dim
            else:
                self.bg_dir_enc = FrequencyEncoder(
                    pe_dim=4, include_input=True, use_pi=False
                )
                in_ch_dir = self.bg_dir_enc.out_dim

            self.bg_hidden_dim = int(bg_hidden)
            self.bg_mlp = nn.Sequential(
                nn.Linear(in_ch_dir, self.bg_hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.bg_hidden_dim, 3, bias=True),
                nn.Sigmoid(),  # keep sigmoid inside like before
            )
            # Lightweight init for the final linear (replicates your _init_bg_head behavior)
            with torch.no_grad():
                final_linear = self.bg_mlp[2]  # the second Linear
                final_linear.weight.mul_(0.1)
                if final_linear.bias is not None:
                    final_linear.bias.zero_()

    # ============================== Routing / Forward / Color / Density ==============================

    def _routing(
        self, pts: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        NO-BATCH routing.
        Soft (boundary_margin > 1): return masked inverse-distance weights (N, K).
        Hard (==1): return argmin assignment (N,).

        Args:
            pts: (N, 3) world-space positions.

        Returns:
            (weights, hard_assign):
                - Soft: (weights (N,K), None)
                - Hard: (None, hard_assign (N,))
        """
        assert pts.dim() == 2 and pts.shape[-1] == 3, "pts must be (N,3)"
        N = pts.shape[0]
        K = self.centroids.shape[0]
        assert K > 0, "No centroids provided."

        # Align device/dtype; compute distances in fp32 for stability, cast back as needed.
        x_pos = pts.to(
            device=self.centroids.device, dtype=self.centroids.dtype
        )  # (N,3)
        c_pos = self.centroids[:, :3].to(
            device=x_pos.device, dtype=x_pos.dtype
        )  # (K,3)

        # Select routing coordinates via your index list (e.g., [1,2] for YZ or [0,1,2] for XYZ)
        x_cluster = x_pos[:, self._coord_idx]  # (N, d)
        c_cluster = c_pos[:, self._coord_idx]  # (K, d)

        dist = torch.cdist(x_cluster.float(), c_cluster.float())  # (N, K), fp32
        if self.boundary_margin > 1:
            dist = dist.clamp_min(1e-6)
            invd = 1.0 / dist  # (N, K)
            mind = dist.min(dim=1, keepdim=True).values  # (N, 1)
            mask = dist <= (self.boundary_margin * mind)  # (N, K)
            invd = invd * mask
            denom = invd.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = (invd / denom).to(
                dtype=x_pos.dtype, device=x_pos.device
            )  # (N, K)
            return weights, None

        hard_assign = dist.argmin(dim=1).to(device=x_pos.device)  # (N,)
        return None, hard_assign

    def forward(
        self,
        x: torch.Tensor,  # (N, D), D >= 6 in your use
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        """
        NO-BATCH mefaNeRF-style sparse expert execution with YOUR semantics:
        - Experts get FULL x[sel] (no xyz stripping).
        - Hard: owner overwrites. Soft: weighted add on selected indices only.
        """
        assert x.dim() == 2 and x.shape[-1] >= 6, "x must be (N,D>=6)"
        N, D = x.shape
        K = len(self.submodules)

        # Per-expert params (optional)
        if params is not None:
            sub_params = [self.get_subdict(params, f"submodules.{k}") for k in range(K)]
        else:
            sub_params = [None] * K

        if active_module is not None:
            sub = self.submodules[active_module]
            y = sub(x, params=sub_params[active_module])  # (N, dim_out)
            return y

        # Route once (no grad) using xyz = x[:, :3]
        with torch.no_grad():
            weights, hard = self._routing(x[:, :3])  # (N,K) or (N,)

        # Lazy init: infer output dim from first non-empty expert
        results = None

        if weights is not None:
            # Soft routing: accumulate weighted outputs where weight > 0
            for k, sub in enumerate(self.submodules):
                wk = weights[:, k]  # (N,)
                sel = (wk > 0).nonzero(as_tuple=False).squeeze(1)  # (n_k,)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub(x[:0], params=sub_params[k])  # touch expert
                    continue

                xk = x.index_select(0, sel)  # (n_k, D)
                yk = sub(xk, params=sub_params[k])  # (n_k, dim_out)

                if results is None:
                    results = x.new_zeros(N, yk.shape[-1])

                results.index_add_(0, sel, yk * wk.index_select(0, sel).unsqueeze(1))
        else:
            # Hard routing: owner overwrites its region
            for k, sub in enumerate(self.submodules):
                sel = (hard == k).nonzero(as_tuple=False).squeeze(1)  # (n_k,)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub(x[:0], params=sub_params[k])
                    continue

                xk = x.index_select(0, sel)  # (n_k, D)
                yk = sub(xk, params=sub_params[k])  # (n_k, dim_out)

                if results is None:
                    results = x.new_zeros(N, yk.shape[-1])

                results.index_copy_(0, sel, yk)

        if results is None:
            # Degenerate case: no expert selected any point
            dim_out = getattr(self, "dim_out", x.shape[-1])
            results = x.new_zeros(N, dim_out)

        return results

    def color(
        self,
        xyz: torch.Tensor,  # (N,3)
        dirs: torch.Tensor,  # (N,3)
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        """
        NO-BATCH color-only query.
        Soft routing blends plain rgb with routing weights (σ-unaware by design here).
        Returns: (N,3)
        """
        assert xyz.dim() == 2 and xyz.shape[-1] == 3, "xyz must be (N,3)"
        assert dirs.dim() == 2 and dirs.shape[-1] == 3, "dirs must be (N,3)"
        if dirs.device != xyz.device:
            dirs = dirs.to(xyz.device)
        dirs = F.normalize(dirs, dim=-1)
        N = xyz.shape[0]
        K = len(self.submodules)

        # per-expert params (optional)
        if params is not None:
            sub_params = [self.get_subdict(params, f"submodules.{k}") for k in range(K)]
        else:
            sub_params = [None] * K

        if active_module is not None:
            sub = self.submodules[active_module]
            dens_out = sub.density(
                xyz, params=sub_params[active_module], return_feats=True
            )
            rgb = sub.color(
                dirs, dens_out["geo_feat"], params=sub_params[active_module]
            )
            return rgb  # (N,3)

        with torch.no_grad():
            weights, hard_assign = self._routing(xyz)  # (N,K) or (N,)

        results = xyz.new_zeros(N, 3)

        if weights is not None:
            # soft: rgb_mix = Σ_k w_k * rgb_k  on indices where w_k>0
            for k, sub in enumerate(self.submodules):
                wk = weights[:, k]  # (N,)
                sel = (wk > 0).nonzero(as_tuple=False).squeeze(1)  # (n_k,)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub.density(
                            xyz[:0], params=sub_params[k], return_feats=True
                        )
                    continue
                sub_xyz = xyz.index_select(0, sel)  # (n_k,3)
                sub_dirs = dirs.index_select(0, sel)  # (n_k,3)
                dens_out = sub.density(sub_xyz, params=sub_params[k], return_feats=True)
                rgb_loc = sub.color(
                    sub_dirs, dens_out["geo_feat"], params=sub_params[k]
                )  # (n_k,3)
                results.index_add_(
                    0, sel, rgb_loc * wk.index_select(0, sel).unsqueeze(1)
                )
            return results

        # hard: copy owner rgb
        for k, sub in enumerate(self.submodules):
            sel = (hard_assign == k).nonzero(as_tuple=False).squeeze(1)  # (n_k,)
            if sel.numel() == 0:
                if self.joint_training:
                    _ = sub.density(xyz[:0], params=sub_params[k], return_feats=True)
                continue
            sub_xyz = xyz.index_select(0, sel)
            sub_dirs = dirs.index_select(0, sel)
            dens_out = sub.density(sub_xyz, params=sub_params[k], return_feats=True)
            rgb_loc = sub.color(
                sub_dirs, dens_out["geo_feat"], params=sub_params[k]
            )  # (n_k,3)
            results.index_copy_(0, sel, rgb_loc)
        return results

    def density(
        self,
        xyz: torch.Tensor,  # (N,3)
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        """
        NO-BATCH density-only query.
        Soft routing: σ_mix = Σ_k w_k σ_k  on the selected indices.
        Hard routing: copy σ from the owner expert.
        Returns: (N,)  (change to (N,1) if your callers expect an extra dim)
        """
        assert xyz.dim() == 2 and xyz.shape[-1] == 3, "xyz must be (N,3)"
        N = xyz.shape[0]
        K = len(self.submodules)

        if params is not None:
            sub_params = [self.get_subdict(params, f"submodules.{k}") for k in range(K)]
        else:
            sub_params = [None] * K

        if active_module is not None:
            sub = self.submodules[active_module]
            sigma = sub.density(xyz, params=sub_params[active_module])  # (N,) or (N,1)
            sigma = sigma.to(device=xyz.device, dtype=xyz.dtype)
            return sigma.squeeze(-1)  # (N,)

        with torch.no_grad():
            weights, hard_assign = self._routing(xyz)  # (N,K) or (N,)

        sigmas = xyz.new_zeros(N)  # (N,)

        if weights is not None:
            # soft: weighted accumulation on indices with w_k>0
            for k, sub in enumerate(self.submodules):
                wk = weights[:, k]  # (N,)
                sel = (wk > 0).nonzero(as_tuple=False).squeeze(1)  # (n_k,)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub.density(xyz[:0], params=sub_params[k])
                    continue
                sub_xyz = xyz.index_select(0, sel)  # (n_k,3)
                sigma_k = sub.density(
                    sub_xyz, params=sub_params[k]
                )  # (n_k,) or (n_k,1)
                sigma_k = sigma_k.to(device=xyz.device, dtype=xyz.dtype).squeeze(
                    -1
                )  # (n_k,)
                sigmas.index_add_(0, sel, sigma_k * wk.index_select(0, sel))
            return sigmas  # (N,)

        # hard: copy owner σ
        for k, sub in enumerate(self.submodules):
            sel = (hard_assign == k).nonzero(as_tuple=False).squeeze(1)
            if sel.numel() == 0:
                if self.joint_training:
                    _ = sub.density(xyz[:0], params=sub_params[k])
                continue
            sub_xyz = xyz.index_select(0, sel)
            sigma_k = sub.density(sub_xyz, params=sub_params[k])  # (n_k,) or (n_k,1)
            sigma_k = sigma_k.to(device=xyz.device, dtype=xyz.dtype).squeeze(
                -1
            )  # (n_k,)
            sigmas.index_copy_(0, sel, sigma_k)
        return sigmas

    # ------------------------------- Background -------------------------------

    def background_color(self, d: torch.Tensor) -> torch.Tensor:

        if not self.use_bg_nerf:
            raise RuntimeError("background_color called but use_bg_nerf=False")

        # normalize
        if d.dim() == 2:
            dn = F.normalize(d, dim=-1)
            enc = self.bg_dir_enc(dn)
            # align dtype/device with bg_mlp first Linear
            first_linear = self.bg_mlp[0]  # nn.Linear
            enc = enc.to(
                dtype=first_linear.weight.dtype, device=first_linear.weight.device
            )
            return self.bg_mlp(enc)
        elif d.dim() == 3:
            B, N, _ = d.shape
            dn = F.normalize(d, dim=-1).reshape(-1, 3)
            enc = self.bg_dir_enc(dn)
            first_linear = self.bg_mlp[0]
            enc = enc.to(
                dtype=first_linear.weight.dtype, device=first_linear.weight.device
            )
            rgb = self.bg_mlp(enc)
            return rgb.view(B, N, 3)
        else:
            raise ValueError(
                f"background_color expects (N,3) or (B,N,3), got {tuple(d.shape)}"
            )

    def maybe_update_expert_occupancies(self, step, params=None):
        for sub in self.submodules:
            sub.maybe_update_occ_grid(step, params)

    def freeze_exert_occupancies(self, flag: bool):
        for sub in self.submodules:
            sub.occ_frozen = flag

    @torch.no_grad()
    def premark_invisible_expert_cells(
        self,
        metas: List[ImageMetadata],
        near_plane: float = 0.0,
        chunk: int = 32**3,
    ) -> List[int]:
        """
        One-time visibility pruning for ALL experts. Call exactly once before training.

        Args:
          mds_by_expert: a list of the metadata of all the cameras
          near_plane: near plane used by ray sampling.
          chunk: chunk size forwarded to occ_grid.mark_invisible_cells.
          quiet: suppress per-expert prints.

        Returns:
          marked_counts: list[int], number of invisible cells marked per expert (for logging).

        """
        if self.cells_premarked or not self.use_occ:
            return [0] * len(self.submodules)

        marked_counts: List[int] = []
        total_counts: List[int] = []
        for k, expert in enumerate(self.submodules):
            expert.premark_invisible_cells(
                metas,
                near_plane=float(near_plane),
                chunk=chunk,
            )

            # Count marked (optional; safe if occs exists)
            n_marked = (
                int((expert.occ_grid.occs < 0).sum().item())
                if hasattr(expert.occ_grid, "occs")
                else 0
            )
            
            
            marked_counts.append(n_marked)
            total_counts.append(expert.occ_grid.occs.numel() if hasattr(expert.occ_grid, "occs") else 0)
            print(
                f"[OCC] container: expert#{k} cams={len(metas)} marked_invisible={n_marked} out of {total_counts[-1]} cells. Invisible ratio: {100.0 * n_marked / total_counts[-1]:.2f}%"
            )
        print("[OCC] container: premark complete for all experts.")
        return marked_counts

    @property
    def occ_ready(self):
        return all(sub.occ_ready for sub in self.submodules)

    @property
    def cells_premarked(self):
        return all(sub.occ_premarked for sub in self.submodules)

    def _init_bg_head(self) -> None:
        """Initialize the final BG linear a bit smaller (your previous behavior)."""
        if not getattr(self, "bg_mlp", None):
            return
        with torch.no_grad():
            # Prefer explicit "out"; fallback to last submodule if renamed
            mod = getattr(self.bg_mlp, "_modules", {})
            last = mod.get("out", None) or (
                mod[list(mod)[-1]] if len(mod) > 0 else None
            )
            lin = getattr(last, "linear", None)
            if lin is not None:
                lin.weight.mul_(0.1)
                if lin.bias is not None:
                    lin.bias.zero_()

    def get_param_groups(self) -> Dict[str, Dict]:
        """
        Aggregate param groups from all experts + background.

        Global groups:
          - 'encoding': all expert encoders (e.g. hash grids / PE)
          - 'sigma':    all expert density / geometry MLPs
          - 'color':    all expert color MLPs
          - 'background': bg_dir_enc + bg_mlp (if use_bg_nerf=True)

        Returns:
            Dict[str, Dict]: e.g.
                {
                    "encoding":   {"params": [...]},
                    "sigma":      {"params": [...]},
                    "color":      {"params": [...]},
                    "background": {"params": [...]},
                }
        """
        encoding_params: List[torch.nn.Parameter] = []
        sigma_params:    List[torch.nn.Parameter] = []
        color_params:    List[torch.nn.Parameter] = []
        bg_params:       List[torch.nn.Parameter] = []

        # ----- collect from experts -----
        for sub in self.submodules:
            if hasattr(sub, "get_param_groups"):
                sub_groups = sub.get_param_groups()
                # encoding
                enc_group = sub_groups.get("encoding", None)
                if enc_group is not None and "params" in enc_group:
                    encoding_params.extend(list(enc_group["params"]))
                # sigma
                sig_group = sub_groups.get("sigma", None)
                if sig_group is not None and "params" in sig_group:
                    sigma_params.extend(list(sig_group["params"]))
                # color
                col_group = sub_groups.get("color", None)
                if col_group is not None and "params" in col_group:
                    color_params.extend(list(col_group["params"]))
            else:
                # Fallback: dump all params into sigma group (or a dedicated 'experts' group)
                sigma_params.extend(list(sub.parameters()))

        # ----- background -----
        if self.use_bg_nerf:
            # bg_dir_enc may or may not have parameters depending on implementation.
            if hasattr(self, "bg_dir_enc"):
                bg_params.extend(list(self.bg_dir_enc.parameters()))
            if hasattr(self, "bg_mlp"):
                bg_params.extend(list(self.bg_mlp.parameters()))

        # ----- build final dict (skip empty groups) -----
        groups: Dict[str, Dict] = {}

        if len(encoding_params) > 0:
            groups["encoding"] = {"params": encoding_params}
        if len(sigma_params) > 0:
            groups["sigma"] = {"params": sigma_params}
        if len(color_params) > 0:
            groups["color"] = {"params": color_params}
        if len(bg_params) > 0:
            groups["background"] = {"params": bg_params}
        return groups    