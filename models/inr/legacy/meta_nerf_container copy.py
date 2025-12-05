from typing import Optional, OrderedDict, Literal, Tuple, Dict
import nerfacc
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.metamodule import MetaModule, MetaLayerBlock
from models.inr.legacy.meta_nerf import MetaNeRFInstant, MetaNeRFStandard
from models.encodings import SHEncoder, FrequencyEncoder, HashGridEncoder


def build_nerf(nerf_variant, **nerf_kwargs):
    if nerf_variant == "instant":
        return MetaNeRFInstant(**nerf_kwargs)
    return MetaNeRFStandard(**nerf_kwargs)


class MetaContainer(MetaModule):
    """
    Container:
      - Owns routing & background.
      - Owns a GLOBAL coarse occupancy grid + PER-EXPERT fine occupancy grids.
      - Optionally provides a SHARED LOW-FREQ hash encoder for experts to concatenate
        with their private HIGH-FREQ encoders (experts implement the concat).

    Additions in this version:
      - set_stage("inner"/"outer") to freeze occ updates during inner meta loop.
      - set_render_step_size(Δ) to keep α = 1 - exp(-σΔ) consistent with marching.
      - mark_invisible_cells_all(...) one-shot camera culling for global+experts.
      - maybe_update_occupancy(step) single entry-point; cadence kept in attributes.
      - Occ threshold ramp helper to avoid over-pruning early.
    """

    def __init__(
        self,
        num_submodules: int,
        centroids: torch.Tensor,
        aabb: torch.Tensor,
        nerf_variant: Literal["instant", "standard"] = "instant",
        *,
        # ---- routing options ----
        boundary_margin: float = 1.0,
        cluster_2d: bool = True,
        joint_training: bool = False,
        # ---- background head ----
        use_bg_nerf: bool = True,
        bg_hidden: int = 32,
        bg_encoder: Literal["spherical", "fourier"] = "spherical",
        # ---- occupancy (two-tier) ----
        use_occ: bool = True,
        occ_conf: Optional[Dict] = None,
        # ---- shared low-freq hash (global prior) ----
        use_shared_lowfreq_hash: bool = True,
        shared_lowfreq_conf: Optional[Dict] = None,
        # ---- forwarded to each expert (hidden sizes, etc.) ----
        **nerf_kwargs,
    ):
        super().__init__()
        assert num_submodules > 0
        assert centroids.ndim == 2 and centroids.size(0) == num_submodules
        assert boundary_margin >= 1.0

        # World-space AABB
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

        # === NEW: training stage & render step size (Δ)
        self.training_stage: Literal["inner", "outer"] = "inner"  # default safe
        self.render_step_size: Optional[float] = None

        # ------------------------------- Shared LOW-FREQ hash -------------------------------
        self.shared_lowfreq_hash: Optional[HashGridEncoder] = None
        if nerf_variant == "instant" and use_shared_lowfreq_hash:
            cfg = shared_lowfreq_conf or {}
            self.shared_lowfreq_hash = HashGridEncoder(aabb=aabb, **cfg)

        # ------------------------------- Experts -------------------------------
        expert_box_list = nerf_kwargs.pop("expert_box_list")

        expert_kwargs_base = {
            **nerf_kwargs,
            "use_occ": use_occ,
            "encoder_xyz_shared_low": self.shared_lowfreq_hash,  # shared low-freq (global AABB), as you wanted
        }

        self.submodules = nn.ModuleList()
        for i, box in enumerate(expert_box_list):
            expert_aabb = box.aabb if hasattr(box, "aabb") else box
            expert_aabb = expert_aabb.detach()

            # Per-expert merged kwargs
            kw = {
                **expert_kwargs_base,
                "aabb": expert_aabb,
            }

            self.submodules.append(build_nerf(nerf_variant, **kw))

        # ------------------------------- Background -------------------------------
        self.use_bg_nerf = bool(use_bg_nerf)
        if self.use_bg_nerf:
            if bg_encoder == "spherical":
                self.bg_dir_enc = SHEncoder(degree=4)
                in_ch_dir = self.bg_dir_enc.out_dim
            else:
                self.bg_dir_enc = FrequencyEncoder(
                    pe_dim=4, include_input=True, use_pi=False
                )
                in_ch_dir = self.bg_dir_enc.out_dim
            self.bg_hidden_dim = int(bg_hidden)
            self.background_head_hidden = MetaLayerBlock(
                in_ch_dir, self.bg_hidden_dim, activation="relu"
            )
            self.background_head_out = MetaLayerBlock(
                self.bg_hidden_dim, 3, activation="sigmoid"
            )
            self._init_bg_head()

        # ------------------------------- Occupancy -------------------------------
        self.use_occ = bool(use_occ)
        self.occ_thre = 0.01
        # cadence & EMA defaults (can be overridden by trainer via attributes)
        diag = torch.linalg.norm(aabb[1] - aabb[0])
        self.occ_params = {
            "warmup": {  # aggressive learning of the grid
                "global_update_period": 1,
                "expert_update_period": 1,
                # EMA=1.0 → no smoothing (fastest to react); if flickery, try 0.997–0.995
                "global_ema_decay": 1.0,
                "expert_ema_decay": 1.0,
                "render_step_size": float((diag / 256.0).item()),
            },
            "steady": {  # stable maintenance
                "global_update_period": 16,
                "expert_update_period": 8,
                "global_ema_decay": 0.99,
                "expert_ema_decay": 0.93,
                "render_step_size": float((diag / 512.0).item()),
            },
        }
        self._apply_occ_phase("warmup")
        self.occ_thre_target = 0.03
        self.occ_warmup_steps = 256  # if >0, cosine-ramp to occ_thre_target
        self.occ_ready = False
        if self.use_occ:
            self._setup_occupancy_grid(occ_conf or {}, aabb, num_submodules)
            # simple counters for lazy updates (optional; harmless if unused)
            self.global_last_update_step = -(10**9)
            self.expert_last_update_step = [-(10**9)] * num_submodules
            self.expert_recent_activity = [0] * num_submodules

    # ============================== NEW: public helpers ==============================

    def _apply_occ_phase(self, phase: Literal["warmup", "steady"]) -> None:
        p = self.occ_params[phase]
        self.global_update_period = int(p["global_update_period"])
        self.expert_update_period = int(p["expert_update_period"])
        self.global_ema_decay = float(p["global_ema_decay"])
        self.expert_ema_decay = float(p["expert_ema_decay"])
        self.render_step_size = float(p["render_step_size"])  # <-- no trailing comma
        self.occ_phase = phase

    def set_stage(self, stage: Literal["inner", "outer"]) -> None:
        """Freeze occupancy updates during inner-loop meta adaptation."""
        self.training_stage = stage

    def set_render_step_size(self, step: float) -> None:
        """Ensure α = 1 - exp(-σΔ) uses the same Δ as the ray marcher."""
        self.render_step_size = float(step)

    @torch.no_grad()
    def mark_invisible_cells_all(
        self,
        K: torch.Tensor,
        c2w: torch.Tensor,
        W: int,
        H: int,
        near: float = 0.0,
    ) -> None:
        """One-time camera visibility culling for global + all experts."""
        if not self.use_occ:
            return
        self.occ_global.mark_invisible_cells(K, c2w, W, H, near)
        for grid in self.occ_expert:
            grid.mark_invisible_cells(K, c2w, W, H, near)

    def record_expert_activity(self, expert_id: int, n_samples: int) -> None:
        """Optional: inform the container an expert processed some rays since last update."""
        if 0 <= expert_id < len(self.expert_recent_activity):
            self.expert_recent_activity[expert_id] += int(n_samples)

    @torch.no_grad()
    def maybe_update_occupancy(self, step: int) -> None:
        """
        Call this ONCE per OUTER step from the trainer. It will:
          - do nothing during inner stage,
          - update global grid on its cadence,
          - optionally update experts that saw rays recently.
        """
        if (not self.occ_ready) and (step >= self.occ_warmup_steps):
            self.occ_ready = True
            self._apply_occ_phase("steady")

        if not self.use_occ or self.training_stage == "inner":
            return

        # Global cadence
        # print(f"last update step {step}, global last update step {self.global_last_update_step}, global_update_period {self.global_update_period}")
        if step - self.global_last_update_step >= self.global_update_period:
            # print("Update is called properly")
            self.update_occupancy_global(
                step,
                warmup_steps=self.occ_warmup_steps,
                ema_decay=self.global_ema_decay,
                every_n=self.global_update_period,
            )
            self.global_last_update_step = step

        # Expert cadence (lazy)
        did_any = False
        for k, last in enumerate(self.expert_last_update_step):
            if (
                step - last
            ) >= self.expert_update_period and self.expert_recent_activity[k] > 0:
                self.update_occupancy_per_expert(
                    step,
                    warmup_steps=self.occ_warmup_steps,
                    ema_decay=self.expert_ema_decay,
                    every_n=self.expert_update_period,
                )
                did_any = True
                break  # next outer step will pick remaining experts
        if did_any:
            # reset handled inside update_occupancy_per_expert
            pass

    # ============================== Occupancy internals ==============================

    def _setup_occupancy_grid(self, occ_conf: Dict, aabb: torch.Tensor, K: int):
        res = int(occ_conf.get("occ_grid_resolution", 128))
        lvls = int(occ_conf.get("occ_grid_levels", 4))
        self.occ_thre = float(occ_conf.get("occ_thre", 0.01))

        # Global coarse grid
        self.occ_global = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb_vec, resolution=res, levels=lvls
        )
        # Per-expert fine grids
        self.occ_expert: nn.ModuleList = nn.ModuleList(
            [
                nerfacc.OccGridEstimator(
                    roi_aabb=self.scene_aabb_vec, resolution=res, levels=lvls
                )
                for _ in range(K)
            ]
        )

    @torch.no_grad()
    def occ_active_fraction(grid) -> float:
        """
        Compute fraction of active (occupied) cells in a nerfacc.OccGridEstimator.
        """
        bins = grid.binaries
        if bins.dtype == torch.bool:
            # Unpacked bool mask
            active = bins.float().mean().item()
        else:
            # Packed uint8 bitfield
            # Use a lookup table to count bits efficiently
            lut = torch.tensor(
                [bin(i).count("1") for i in range(256)],
                device=bins.device,
                dtype=torch.int32,
            )
            bit_count = lut[bins].sum().item()
            active = bit_count / (bins.numel() * 8)
        return float(active)

    @torch.no_grad()
    def update_occupancy_global(
        self,
        step: int,
        *,
        ema_decay: Optional[float] = None,
        warmup_steps: int = 50,
        every_n: Optional[int] = None,
    ):
        """Update GLOBAL grid from blended σ; no-op during inner stage."""
        if not self.use_occ or self.training_stage == "inner":
            return
        ema = self.global_ema_decay if ema_decay is None else ema_decay
        n = self.global_update_period if every_n is None else every_n
        occ_thre = self._current_occ_threshold(step)

        def _alpha_eval(x: torch.Tensor) -> torch.Tensor:
            with torch.cuda.amp.autocast(enabled=False):
                sigma = self._sigma_blended_at_points(x.float())  # (N,)
                return self._alpha_from_sigma(
                    sigma, float(self.render_step_size)
                ).unsqueeze(-1)

        # print("Update every Global!")
        self.occ_global.update_every_n_steps(
            step=step,
            occ_eval_fn=_alpha_eval,
            occ_thre=occ_thre,
            ema_decay=ema,
            warmup_steps=warmup_steps,
            n=n,
        )

    @torch.no_grad()
    def update_occupancy_per_expert(
        self,
        step: int,
        *,
        ema_decay: Optional[float] = None,
        warmup_steps: int = 50,
        every_n: Optional[int] = None,
    ):
        """Update PER-EXPERT grids with hard/soft routing; no-op during inner stage."""
        if not self.use_occ or self.training_stage == "inner":
            return
        ema = self.expert_ema_decay if ema_decay is None else ema_decay
        n = self.expert_update_period if every_n is None else every_n
        occ_thre = self._current_occ_threshold(step)

        def _alpha_eval_k(k: int):
            def fn(x: torch.Tensor) -> torch.Tensor:
                # x: (N, 3) world-space points
                with torch.cuda.amp.autocast(enabled=False):
                    # Route once for the provided points
                    weights, hard = self._routing(
                        x.unsqueeze(0)
                    )  # (1,N,K) or hard: (1,N)
                    if weights is not None:
                        wk = weights[0, :, k]  # (N,)
                        m = wk > 0
                    else:
                        m = hard[0] == k  # (N,)

                    if not m.any():
                        return x.new_zeros(x.shape[0], 1)

                    sigma_k = self._sigma_expert_at_points(k, x[m])  # (n,)
                    alpha_k = self._alpha_from_sigma(
                        sigma_k, float(self.render_step_size)
                    ).unsqueeze(
                        -1
                    )  # (n,1)

                    out = x.new_zeros(x.shape[0], 1)
                    out[m] = alpha_k
                    return out

            return fn

        for k in range(len(self.submodules)):
            # If activity tracking is used, skip idle experts.
            if (
                hasattr(self, "expert_recent_activity")
                and self.expert_recent_activity[k] <= 0
            ):
                continue

            self.occ_expert[k].update_every_n_steps(
                step=step,
                occ_eval_fn=_alpha_eval_k(k),
                occ_thre=occ_thre,
                ema_decay=ema,
                warmup_steps=warmup_steps,
                n=n,
            )
            if hasattr(self, "expert_last_update_step"):
                self.expert_last_update_step[k] = step
            if hasattr(self, "expert_recent_activity"):
                self.expert_recent_activity[k] = 0  # reset after an update

    # ---- σ/α evaluators used by occupancy updates ----
    @torch.no_grad()
    def _sigma_expert_at_points(self, k: int, x: torch.Tensor) -> torch.Tensor:
        """σ from expert k at world points x: (N,3) -> (N,)"""
        sub = self.submodules[k]
        s = sub.density(x)  # dict with "sigma" or raw tensor
        sigma = s["sigma"] if isinstance(s, dict) else s
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        return sigma.squeeze(-1).reshape(-1)

    @torch.no_grad()
    def _sigma_blended_at_points(self, x: torch.Tensor) -> torch.Tensor:
        """Route x across experts (soft or hard), blend σ accordingly."""
        B = 1
        weights, hard = self._routing(x.view(B, -1, 3))
        N = x.shape[0]
        sigma = x.new_zeros(N)

        if weights is not None:
            for k in range(len(self.submodules)):
                w = weights[0, :, k]  # (N,)
                m = w > 0
                if m.any():
                    sigma_k = self._sigma_expert_at_points(k, x[m])  # (n,)
                    sigma[m] += sigma_k * w[m]
        else:
            for k in range(len(self.submodules)):
                m = hard[0] == k
                if m.any():
                    sigma[m] = self._sigma_expert_at_points(k, x[m])
        return sigma

    @staticmethod
    def _alpha_from_sigma(sigma: torch.Tensor, step: float) -> torch.Tensor:
        return 1.0 - torch.exp(-sigma.clamp_min(0) * step)

    def _current_occ_threshold(self, step: int) -> float:
        """Cosine ramp from min(self.occ_thre, 0.005) to occ_thre_target over warmup."""
        if self.occ_warmup_steps is None or self.occ_warmup_steps <= 0:
            return float(self.occ_thre)
        start = min(float(self.occ_thre), 0.005)
        target = float(self.occ_thre_target)
        t = max(0.0, min(1.0, step / float(self.occ_warmup_steps)))
        # cosine interpolation
        return float(
            target
            - 0.5 * (target - start) * (torch.cos(torch.tensor(t * 3.1415926535)) + 1.0)
        )

    # ============================== Routing / Forward / Color / Density ==============================

    def _routing(
        self, pts: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        x_pos = pts.to(self.centroids.dtype)  # (B,N,3)
        c_pos = self.centroids[:, :3].to(x_pos.device)  # (K,3)

        x_cluster = x_pos[..., self._coord_idx]
        c_cluster = c_pos[..., self._coord_idx]

        x_flat = x_cluster.reshape(-1, x_cluster.shape[-1])  # (B*N,d)
        dist = torch.cdist(x_flat, c_cluster)  # (B*N,K)

        if self.boundary_margin > 1:
            dist = dist.clamp_min(1e-6)
            invd = 1.0 / dist
            mind = dist.min(dim=1, keepdim=True)[0]
            mask = dist <= (self.boundary_margin * mind)
            invd = invd * mask
            denom = invd.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = (invd / denom).reshape(pts.shape[0], pts.shape[1], -1)  # (B,N,K)
            return weights, None
        else:
            hard_assign = dist.argmin(dim=1).reshape(
                pts.shape[0], pts.shape[1]
            )  # (B,N)
            return None, hard_assign

    # def forward(self, x: torch.Tensor, params: Optional[OrderedDict] = None, active_module: Optional[int] = None) -> torch.Tensor:
    #     assert x.dim() in (2, 3) and x.shape[-1] >= 6
    #     single_batch = (x.dim() == 2)
    #     if single_batch:
    #         x = x.unsqueeze(0)
    #     B, N, D = x.shape

    #     if active_module is not None:
    #         sub = self.submodules[active_module]
    #         params_act = self.get_subdict(params, f"submodules.{active_module}") if params is not None else None
    #         y = sub(x, params=params_act)
    #         return y.squeeze(0) if single_batch else y

    #     results = x.new_zeros(B, N, self.dim_out)
    #     with torch.no_grad():
    #         weights, hard_assign = self._routing(x[..., :3])

    #     for k, sub in enumerate(self.submodules):
    #         cluster_mask = (weights[..., k] > 0) if weights is not None else (hard_assign == k)
    #         if not cluster_mask.any():
    #             if self.joint_training:
    #                 _ = sub(x[:, :0], params=self.get_subdict(params, f"submodules.{k}") if params is not None else None)
    #             continue
    #         b_idx, n_idx = torch.where(cluster_mask)
    #         for b in b_idx.unique():
    #             sel = n_idx[b_idx == b]
    #             sub_x = x[b:b+1, sel]
    #             params_k = self.get_subdict(params, f"submodules.{k}") if params is not None else None
    #             y = sub(sub_x, params=params_k)
    #             if weights is not None:
    #                 w = weights[b:b+1, sel, k:k+1]
    #                 results[b:b+1, sel] += y * w
    #             else:
    #                 results[b:b+1, sel] = y
    #     return results if not single_batch else results.squeeze(0)

    def forward(
        self,
        x: torch.Tensor,  # (N,6) or (B,N,6)
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        assert x.dim() in (2, 3) and x.shape[-1] >= 6
        single_batch = x.dim() == 2
        if single_batch:
            x = x.unsqueeze(0)  # (1,N,6)
        B, N, D = x.shape
        M = B * N

        if active_module is not None:
            sub = self.submodules[active_module]
            params_act = (
                self.get_subdict(params, f"submodules.{active_module}")
                if params is not None
                else None
            )
            y = sub(x, params=params_act)
            return y.squeeze(0) if single_batch else y

        # Routing once (no grad)
        with torch.no_grad():
            weights, hard = self._routing(x[..., :3])  # (B,N,K) or (B,N)

        x_flat = x.reshape(M, D)
        out = x_flat.new_zeros(M, self.dim_out)  # (M,4)

        if weights is not None:
            # soft routing: blend
            W = weights.reshape(M, -1)  # (M,K)
            for k, sub in enumerate(self.submodules):
                wk = W[:, k]  # (M,)
                sel = wk.nonzero(as_tuple=False).squeeze(1)  # indices with weight>0
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub(
                            x[:, :0],
                            params=(
                                self.get_subdict(params, f"submodules.{k}")
                                if params is not None
                                else None
                            ),
                        )
                    continue
                xk = x_flat.index_select(0, sel).unsqueeze(0)  # (1,n_k,D)
                params_k = (
                    self.get_subdict(params, f"submodules.{k}")
                    if params is not None
                    else None
                )
                yk = sub(xk, params=params_k).squeeze(0)  # (n_k,4)
                out.index_add_(0, sel, yk * wk.index_select(0, sel).unsqueeze(1))
        else:
            # hard assignment: copy
            hard_flat = hard.reshape(M)  # (M,)
            for k, sub in enumerate(self.submodules):
                sel = (hard_flat == k).nonzero(as_tuple=False).squeeze(1)  # (n_k,)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub(
                            x[:, :0],
                            params=(
                                self.get_subdict(params, f"submodules.{k}")
                                if params is not None
                                else None
                            ),
                        )
                    continue
                xk = x_flat.index_select(0, sel).unsqueeze(0)  # (1,n_k,D)
                params_k = (
                    self.get_subdict(params, f"submodules.{k}")
                    if params is not None
                    else None
                )
                yk = sub(xk, params=params_k).squeeze(0)  # (n_k,4)
                out.index_copy_(0, sel, yk)

        out = out.view(B, N, self.dim_out)  # (B,N,4)
        return out.squeeze(0) if single_batch else out

    def color(
        self,
        xyz: torch.Tensor,
        dirs: torch.Tensor,
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        assert xyz.dim() in (2, 3) and dirs.dim() == xyz.dim()
        single_batch = xyz.dim() == 2
        if single_batch:
            xyz = xyz.unsqueeze(0)
            dirs = dirs.unsqueeze(0)
        if dirs.device != xyz.device:
            dirs = dirs.to(xyz.device)
        dirs = F.normalize(dirs, dim=-1)
        B, N, _ = xyz.shape

        if active_module is not None:
            sub = self.submodules[active_module]
            params_act = (
                self.get_subdict(params, f"submodules.{active_module}")
                if params is not None
                else None
            )
            dens_out = sub.density(xyz, params=params_act)
            rgb = sub.color(dirs, dens_out["geo_feat"], params=params_act)
            return rgb.squeeze(0) if single_batch else rgb

        results = dirs.new_zeros(B, N, 3)
        with torch.no_grad():
            weights, hard_assign = self._routing(xyz)

        for k, sub in enumerate(self.submodules):
            cluster_mask = (
                (weights[..., k] > 0) if weights is not None else (hard_assign == k)
            )
            if not cluster_mask.any():
                if self.joint_training:
                    _ = sub.density(
                        xyz[:, :0],
                        params=(
                            self.get_subdict(params, f"submodules.{k}")
                            if params is not None
                            else None
                        ),
                    )
                continue
            b_idx, n_idx = torch.where(cluster_mask)
            for b in b_idx.unique():
                sel = n_idx[b_idx == b]
                sub_xyz = xyz[b : b + 1, sel]
                sub_dirs = dirs[b : b + 1, sel]
                params_k = (
                    self.get_subdict(params, f"submodules.{k}")
                    if params is not None
                    else None
                )
                dens_out = sub.density(sub_xyz, params=params_k)
                rgb_local = sub.color(sub_dirs, dens_out["geo_feat"], params=params_k)
                if weights is not None:
                    w = weights[b : b + 1, sel, k : k + 1]
                    results[b : b + 1, sel] += rgb_local * w
                else:
                    results[b : b + 1, sel] = rgb_local
        return results.squeeze(0) if single_batch else results

    def density(
        self,
        xyz: torch.Tensor,
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        assert xyz.dim() in (2, 3)
        single_batch = xyz.dim() == 2
        if single_batch:
            xyz = xyz.unsqueeze(0)
        B, N, _ = xyz.shape
        sigmas = xyz.new_zeros(B, N)

        if active_module is not None:
            sub = self.submodules[active_module]
            params_act = (
                self.get_subdict(params, f"submodules.{active_module}")
                if params is not None
                else None
            )
            s = sub.density(xyz, params=params_act)
            sigma = s["sigma"] if isinstance(s, dict) else s
            sigma = torch.as_tensor(sigma, device=xyz.device, dtype=xyz.dtype).squeeze(
                -1
            )
            # shape align: (B,N)
            if sigma.dim() == 1:
                sigma = sigma.view(1, -1)
            sigmas.copy_(sigma)
            return sigmas.squeeze(0) if single_batch else sigmas

        with torch.no_grad():
            weights, hard_assign = self._routing(xyz)

        for k, sub in enumerate(self.submodules):
            cluster_mask = (
                (weights[..., k] > 0) if weights is not None else (hard_assign == k)
            )
            if not cluster_mask.any():
                if self.joint_training:
                    _ = sub.density(
                        xyz[:, :0],
                        params=(
                            self.get_subdict(params, f"submodules.{k}")
                            if params is not None
                            else None
                        ),
                    )
                continue
            b_idx, n_idx = torch.where(cluster_mask)
            for b in b_idx.unique():
                sel = n_idx[b_idx == b]
                sub_xyz = xyz[b : b + 1, sel]
                params_k = (
                    self.get_subdict(params, f"submodules.{k}")
                    if params is not None
                    else None
                )
                s = sub.density(sub_xyz, params=params_k)
                sigma_local = s["sigma"] if isinstance(s, dict) else s
                sigma_local = (
                    torch.as_tensor(sigma_local, device=xyz.device, dtype=xyz.dtype)
                    .squeeze(-1)
                    .view(1, -1)
                )
                if weights is not None:
                    w = weights[b : b + 1, sel]  # (1,n)
                    sigmas[b : b + 1, sel] += sigma_local * w
                else:
                    sigmas[b : b + 1, sel] = sigma_local
        return sigmas.squeeze(0) if single_batch else sigmas

    # ------------------------------- Background -------------------------------

    def background_color(
        self, d: torch.Tensor, params: Optional[OrderedDict] = None
    ) -> torch.Tensor:
        bg_enc_params = self._subdict_or_none(params, "bg_dir_enc")
        bg_hid_params = self._subdict_or_none(params, "background_head_hidden")
        bg_out_params = self._subdict_or_none(params, "background_head_out")

        if d.dim() == 2:
            dn = F.normalize(d, dim=-1)
            enc = (
                self.bg_dir_enc(dn)
                if bg_enc_params is None
                else self.bg_dir_enc(dn, params=bg_enc_params)
            )
            h = self.background_head_hidden(enc, params=bg_hid_params)
            rgb = self.background_head_out(h, params=bg_out_params)
            return rgb
        else:
            B, N, _ = d.shape
            dn = F.normalize(d, dim=-1).reshape(-1, 3)
            enc = (
                self.bg_dir_enc(dn)
                if bg_enc_params is None
                else self.bg_dir_enc(dn, params=bg_enc_params)
            )
            h = self.background_head_hidden(enc, params=bg_hid_params)
            rgb = self.background_head_out(h, params=bg_out_params)
            return rgb.view(B, N, 3)

    def _init_bg_head(self) -> None:
        with torch.no_grad():
            lin = getattr(self.background_head_out, "linear", None)
            if lin is not None:
                lin.weight.mul_(0.1)
                if lin.bias is not None:
                    lin.bias.zero_()

    def _subdict_or_none(self, params, module_name: str):
        if params is None:
            return None
        if not any(k.startswith(module_name + ".") for k in params.keys()):
            return None
        return self.get_subdict(params, module_name)
