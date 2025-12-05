import math
from typing import List, Optional, Literal, Dict, Union


import torch
from torch import Tensor

from nerfacc import OccGridEstimator

from data.image_metadata import ImageMetadata
from models.metamodule import MetaModule, MetaLinear, MetaLayerBlock, MetaSequential
from models.trunc_exp import trunc_exp
from models.encodings import FrequencyEncoder, HashGridEncoder, SHEncoder
from nerfs.scene_box import SceneBox


class MetaNGP(MetaModule):
    def __init__(
        self,
        *,
        occ_conf: Dict,
        scene_box: SceneBox,  # (2,3) world AABB
        hidden: int = 64,
        sigma_depth: int = 2,  # <-- interpreted as # of HIDDEN layers
        color_hidden: int = 64,
        geo_feat_dim: int = 15,
        color_depth: int = 3,  # <-- interpreted as # of HIDDEN layers
        use_sigmoid_rgb: bool = True,
        hash_enc_conf=None,
        dir_encoding: Literal["spherical", "frequency"] = "spherical",
        **kwargs,
    ) -> None:
        super().__init__()
        hash_enc_conf = hash_enc_conf or {}

        self.use_occ = occ_conf.get("use_occ", False)
        self.geo_feat_dim = int(geo_feat_dim)
        self.use_sigmoid_rgb = bool(use_sigmoid_rgb)
        self.scene_box = scene_box
        aabb = scene_box.aabb
        assert isinstance(aabb, torch.Tensor) and aabb.shape == (2, 3)
        self.register_buffer("aabb_extent", scene_box.extent)
        self.register_buffer(
            "enc_eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False
        )

        # -----------------------
        # XYZ encoders (concat)
        # -----------------------
        self.xyz_encoder = HashGridEncoder(
            levels=hash_enc_conf.get("levels", 4),
            min_res=hash_enc_conf.get("min_res", 16),
            max_res=hash_enc_conf.get("max_res", 4096),
            log2_hashmap_size=hash_enc_conf.get("log2_hashmap_size", 19),
            features_per_level=hash_enc_conf.get("features_per_level", 2),
            interpolation=hash_enc_conf.get("interpolation", "Linear"),
        )

        in_ch_xyz = self.xyz_encoder.out_dim

        # -----------------------
        # Direction encoder
        # -----------------------
        dir_encoding = dir_encoding.lower()
        if dir_encoding == "frequency":
            self.dir_encoder = FrequencyEncoder(
                in_dim=3, pe_dim=4, include_input=True, use_pi=False
            )
            in_ch_dir = self.dir_encoder.out_dim
        elif dir_encoding == "spherical":
            self.dir_encoder = SHEncoder(levels=4)
            in_ch_dir = self.dir_encoder.out_dim
        else:
            raise ValueError(f"Unsupported dir_encoding: {dir_encoding}")

        # -----------------------
        # Sigma_net (meta)
        # depth = number of HIDDEN layers
        # -----------------------
        sigma_trunk = []
        last = in_ch_xyz
        for _ in range(
            max(int(sigma_depth), 0)
        ):  # build EXACTLY sigma_depth hidden blocks
            sigma_trunk.append(MetaLayerBlock(last, hidden, activation="relu"))
            last = hidden
        self.sigma_trunk = MetaSequential(*sigma_trunk)

        self.sigma_head = MetaLinear(last, 1)
        with torch.no_grad():
            self.sigma_head.bias.fill_(-1.0)
        self.geo_head = MetaLinear(last, self.geo_feat_dim)
        self.sigma_act = trunc_exp

        # -----------------------
        # Color MLP (meta)
        # depth = number of HIDDEN layers
        # -----------------------
        color_mlp = []
        last = self.geo_feat_dim + in_ch_dir
        for _ in range(
            max(int(color_depth), 0)
        ):  # build EXACTLY color_depth hidden blocks
            color_mlp.append(MetaLayerBlock(last, color_hidden, activation="relu"))
            last = color_hidden

        # final RGB output (always appended)
        color_mlp.append(MetaLinear(last, 3))
        self.color_mlp = MetaSequential(*color_mlp)
        self.rgb_act = (
            torch.nn.Sigmoid() if self.use_sigmoid_rgb else torch.nn.Identity()
        )
        # -----------------------
        # Occupancy handling
        # -----------------------
        if self.use_occ:
            self.render_step_size = (
                float(occ_conf["render_step_size"])
                if "render_step_size" in occ_conf
                and occ_conf["render_step_size"] is not None
                else float(scene_box.get_diagonal_length()) / 1000.0
            )

            self.occ_thre: float = float(occ_conf.get("occ_thre", 1e-2))
            self.alpha_thre: float = float(occ_conf.get("alpha_thre", 1e-2))
            self.cone_angle: float = float(occ_conf.get("cone_angle", 1.0 / 256.0))
            self.near_plane: float = float(occ_conf.get("near_plane", 0.05))
            self.far_plane: float = float(occ_conf.get("far_plane", 1e3))
            self.occ_update_interval: int = int(occ_conf.get("update_interval", 16))
            self.occ_warmup_steps: int = int(occ_conf.get("warmup_steps", 256))
            self.occ_cosine_anneal = bool(occ_conf.get("cosine_anneal", True))
            self.occ_alpha_thre_start: float = float(
                occ_conf.get("alpha_thre_start", 0.0)
            )
            self.occ_alpha_thre_end: float = float(
                occ_conf.get("alpha_thre_end", self.alpha_thre)
            )
            self.occ_ema_decay: float = float(occ_conf.get("ema_decay", 0.95))
            self.occ_resolution: int = int(occ_conf.get("resolution", 128))
            self.occ_levels: int = int(occ_conf.get("levels", 4))
            scene_aabb = torch.cat([self.scene_box.min, self.scene_box.max]).flatten()
            self.register_buffer("scene_aabb", scene_aabb)

            self.occ_grid = OccGridEstimator(
                roi_aabb=self.scene_aabb,
                resolution=self.occ_resolution,
                levels=self.occ_levels,
            )

            self._check_aabb()
            # Phase flags
            self.occ_frozen: bool = bool(occ_conf.get("occ_frozen", False))
            self.occ_ready: bool = bool(occ_conf.get("occ_ready", False))
            self.num_occ_updates = 0
            self.occ_premarked = False

    # -----------------------
    # Utilities
    # -----------------------
    def _check_aabb(self):
        aabb = self.scene_aabb
        assert aabb.dtype == torch.float32 and aabb.isfinite().all()
        assert aabb.numel() == 6
        mn, mx = aabb[:3], aabb[3:]
        if not (mn < mx).all():
            raise ValueError(f"AABB invalid: min>=max ({mn} vs {mx})")
        # Must be on same device as rays/occ_grid
        assert aabb.device == self.occ_grid.aabbs.device

    def _world_to_unit(self, x):
        x01 = (x - self.scene_box.min) / self.aabb_extent
        return x01.clamp(self.enc_eps, 1.0 - self.enc_eps)

    def _enc_xyz(self, x_world: Tensor) -> Tensor:
        """Encode xyz."""
        x01 = self._world_to_unit(x_world)
        return self.xyz_encoder(x01)  # (..., Ch)

    def _enc_dir(self, d: Tensor) -> Tensor:
        """Encode directions with the selected dir encoder (feed unit vectors)."""
        d = d / (
            d.norm(dim=-1, keepdim=True).clamp_min_(1e-9)
        )  # good for both encoders
        return self.dir_encoder(d)

    # -----------------------
    # Forward parts
    # -----------------------
    def density(
        self, x: Tensor, params: Optional[Dict[str, Tensor]] = None, return_feats=False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """xyz encoding → sigma trunk → heads.

        Args:
            x: (...,3) world coords.
            params: optional meta-params (expects keys under 'sigma_trunk', 'sigma_head', 'geo_head').
            return_feats: if True, return dict {'sigma', 'geo_feat'}; else return sigma tensor.

        Returns:
            Tensor: (...,1) sigma if return_feats=False
            or
            Dict[str, Tensor]: {'sigma': (...,1), 'geo_feat': (...,G)} if return_feats=True.
        """
        h = self._enc_xyz(x)  # (..., Cx) with [0,1) normalization inside
        h = self.sigma_trunk(h, params=self.get_subdict(params, "sigma_trunk"))

        sigma_raw = self.sigma_head(
            h, params=self.get_subdict(params, "sigma_head")
        )  # (...,1)
        sigma = self.sigma_act(sigma_raw)  # (...,1), AMP-safe TruncExp
        if not return_feats:
            return sigma  # for nerfacc

        geo_feat = self.geo_head(
            h, params=self.get_subdict(params, "geo_head")
        )  # (..., G)

        return {"sigma": sigma, "geo_feat": geo_feat}

    def color(
        self, d: Tensor, geo_feat: Tensor, params: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """[geo_feat, dir_enc] -> color MLP -> rgb"""

        d_enc = self._enc_dir(d)  # (..., Cd)
        h_rgb = torch.cat([geo_feat, d_enc], dim=-1)  # (..., G+Cd)
        h_rgb = self.color_mlp(h_rgb, params=self.get_subdict(params, "color_mlp"))
        return self.rgb_act(h_rgb)  # (..., 3)

    def forward(self, x_d: torch.Tensor, params=None):
        """
        Forward pass expecting a single input tensor of shape (...,6)
        where x_d = [xyz(3), dir(3)].

        Returns:
            Tensor: (..., 4) concatenation [rgb(3), sigma(1)].
        """
        assert x_d.shape[-1] == 6, f"Expected (...,6) [xyz,dir], got {x_d.shape}"
        x, d = x_d[..., :3], x_d[..., 3:6]

        density = self.density(x, params=params, return_feats=True)
        rgb = self.color(d, density["geo_feat"], params=params)
        return torch.cat(
            [rgb, density["sigma"]], dim=-1
        )  # (...,4) = [rgb(3), sigma(1)]

    # ======================= Occupancy Helpers =======================
    def _anneal_alpha_thre(self, step: int):
        """Cosine/linear ramp from start→end over warmup, then hold."""
        if step < self.occ_warmup_steps:
            # progress in [0,1] over the warmup window
            t = step / max(1, self.occ_warmup_steps - 1)
            if self.occ_cosine_anneal:
                cos = 0.5 * (1 - math.cos(math.pi * t))
                self.alpha_thre = (
                    1 - cos
                ) * self.occ_alpha_thre_start + cos * self.occ_alpha_thre_end
            else:
                self.alpha_thre = (
                    1 - t
                ) * self.occ_alpha_thre_start + t * self.occ_alpha_thre_end
        else:
            self.alpha_thre = self.occ_alpha_thre_end

    @torch.no_grad()
    def _build_intrinsics_from_metadata(
        self, mds: List[ImageMetadata], device: torch.device
    ) -> torch.Tensor:
        """Stack intrinsics into (N,3,3) OpenCV-style K matrices."""
        def _make_K(md: ImageMetadata) -> torch.Tensor:
            Kraw = torch.as_tensor(md.intrinsics, dtype=torch.float32)
            if Kraw.numel() == 9:
                return Kraw.view(3, 3)
            if Kraw.numel() == 4:  # [fx, fy, cx, cy]
                fx, fy, cx, cy = Kraw.unbind()
                return torch.tensor(
                    [[fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )
            raise ValueError(f"Unsupported intrinsics shape: {tuple(Kraw.shape)}")

        K = torch.stack([_make_K(md) for md in mds], dim=0)  # (N,3,3)
        return K.to(device=device, dtype=torch.float32)


    @torch.no_grad()
    def _build_c2w_rdf_from_metadata(
        self, mds: List[ImageMetadata], device: torch.device
    ) -> torch.Tensor:
        """
        Stack c2w from metadata and convert camera basis RUB -> RDF (+Z forward, y down),
        keeping world frame (DRB) unchanged.
        """
        c2w = torch.stack(
            [torch.as_tensor(md.c2w, dtype=torch.float32) for md in mds], dim=0
        )  # (N,3,4) or (N,4,4)

        C3 = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32))  # RUB->RDF

        if c2w.shape[1:] == (3, 4):
            R = c2w[:, :3, :3]
            t = c2w[:, :3, 3:]
            R_rdf = torch.einsum("nij,jk->nik", R, C3)  # (N,3,3)
            c2w_rdf = torch.cat([R_rdf, t], dim=2)      # (N,3,4)
        elif c2w.shape[1:] == (4, 4):
            c2w_rdf = c2w.clone()
            R = c2w_rdf[:, :3, :3]
            R_rdf = torch.einsum("nij,jk->nik", R, C3)
            c2w_rdf[:, :3, :3] = R_rdf
        else:
            raise ValueError(f"Unsupported c2w shape: {tuple(c2w.shape)}")

        return c2w_rdf.to(device=device, dtype=torch.float32)


    # ======================= Occupancy =======================
    @torch.no_grad()
    def premark_invisible_cells(
        self,
        mds: List[ImageMetadata],
        near_plane: float = 0.05,
        chunk: int = 32**3,
    ) -> None:
        """
        One-time visibility pruning for THIS expert's occupancy grid.

        Uses global ImageMetadata:
        - intrinsics -> OpenCV K (N,3,3)
        - c2w (RUB->DRB) -> (RDF(+Z forward,y down)->DRB)
        then calls nerfacc.mark_invisible_cells to mark never-visible cells as occ = -1.
        """
        if not self.use_occ or self.occ_premarked:
            return

        # Filter out Nones defensively
        mds = [md for md in mds if md is not None]
        if len(mds) == 0:
            print("[OCC] premark skipped: empty metadata list.")
            self.occ_premarked = True
            return

        device = self.occ_grid.aabbs.device

        K = self._build_intrinsics_from_metadata(mds, device)
        c2w_rdf = self._build_c2w_rdf_from_metadata(mds, device)

        # All images should share the same H,W after dataset prep
        H, W = int(mds[0].H), int(mds[0].W)

        self.occ_grid.mark_invisible_cells(
            K=K,
            c2w=c2w_rdf,
            width=W,
            height=H,
            near_plane=float(near_plane),
            chunk=chunk,
        )

        self.occ_premarked = True


    @torch.no_grad()
    def maybe_update_occ_grid(
        self, step: int, params: Optional[Dict[str, Tensor]] = None
    ) -> None:
        """
        Maybe update occupancy grid during training.

        Args:
            step (int): global step count.
            params (Optional[Dict[str, Tensor]]): model parameters.
            log_every (int): log every N steps.

        Returns:
            None
        """
        if not (self.training and self.use_occ and not self.occ_frozen):
            return

        self.occ_ready = step >= self.occ_warmup_steps
        # Anneal alpha threshold (used by occupancy_marching)
        self._anneal_alpha_thre(step)

        # Density(midpoints) for occ grid updates, no grads
        def occ_eval_fn(x: Tensor) -> Tensor:
            with torch.no_grad():
                return self.density(x, params=params).squeeze(-1) * self.render_step_size

        # Let nerfacc handle "every n steps" logic internally.
        self.occ_grid.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=self.occ_thre,
            ema_decay=self.occ_ema_decay,
            warmup_steps=self.occ_warmup_steps,
            n=self.occ_update_interval,
        )
        self.num_occ_updates += 1
        if (step % self.occ_update_interval == 0):
            print(f"[OCC UPDATE {self.num_occ_updates}] Step= {step:5d} | Warmup={step < self.occ_warmup_steps} |" 
                  f" α_thre={self.alpha_thre:6.4f} | Interval=" 
                  f"{self.occ_update_interval} | UPDATED")


    @torch.no_grad()
    def occupancy_marching(
        self,
        rays: torch.Tensor,
        *,
        params: Optional[Dict[str, torch.Tensor]] = None,
        render_step_size: Optional[float] = None,
        alpha_thre: Optional[float] = None,
        cone_angle: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Volumetric ray marching using THIS expert's occupancy grid.

        Args:
            rays: (N, 8) [o(3), d(3), t_min, t_max] in the same world frame as the AABB.
            params: optional meta-params for sigma evaluation during training.
            render_step_size, alpha_thre, cone_angle: optional overrides.

        Returns:
            (ray_indices, t_starts, t_ends)
        """
        if getattr(self, "occ_grid", None) is None:
            raise RuntimeError("MetaNGP: occ_grid missing")

        # (N, 3), (N, 3), (N,), (N,)
        rays = rays.contiguous()
        o = rays[:, :3]
        d = rays[:, 3:6]
        t_min = rays[:, 6]
        t_max = rays[:, 7]

        # Training: use sigma_fn for extra skipping; eval: grid-only
        sigma_fn = None
        if self.training:
            def sigma_fn(
                t_starts: torch.Tensor,
                t_ends: torch.Tensor,
                ray_indices: torch.Tensor,
            ) -> torch.Tensor:
                mids = 0.5 * (t_starts + t_ends)                 # (M,)
                x = o[ray_indices] + d[ray_indices] * mids[:, None]  # (M, 3)
                return self.density(x, params=params).squeeze(-1)    # (M,)

        ray_indices, t_starts, t_ends = self.occ_grid.sampling(
            rays_o=o,
            rays_d=d,
            sigma_fn=sigma_fn,
            t_min=t_min,
            t_max=t_max,
            render_step_size=(
                self.render_step_size if render_step_size is None else render_step_size
            ),
            stratified=self.training,
            cone_angle=(
                self.cone_angle if cone_angle is None else cone_angle
            ),
            alpha_thre=(
                self.alpha_thre if alpha_thre is None else alpha_thre
            ),
        )

        return ray_indices, t_starts, t_ends

    def get_param_groups(self) -> Dict[str, Dict]:
        """
        Return optimizer param groups for fine-grained LR / regularization.

        Groups:
            - 'encoding': hash grid / xyz encoder (TCNN-style)
            - 'sigma':    density + geometric features MLP
            - 'color':    color MLP

        The dict values are partial optimizer group specs; you can
        attach lr, weight_decay, etc. in the training script.
        """
        encoding_params = list(self.xyz_encoder.parameters())

        sigma_params = (
            list(self.sigma_trunk.parameters())
            + list(self.sigma_head.parameters())
            + list(self.geo_head.parameters())
        )
        color_params = list(self.color_mlp.parameters())

        return {
            "encoding": {"params": encoding_params},
            "sigma":    {"params": sigma_params},
            "color":    {"params": color_params},
        }
        
   
    