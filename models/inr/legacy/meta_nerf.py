# Clean, container-friendly NeRF modules with optional shared low-freq hash encoder
# and private high-freq hash encoder per expert (Instant-NGP style).
#
# Key design decisions:
# - World-space consistently (pass full-scene AABB to encoders; no unit-box surprises).
# - density(...) returns {"sigma": (...,1), "geo_feat": (...,G)} for container compatibility.
# - MetaNeRFInstant may receive an optional shared low-freq hash encoder from the container.
#   It is NOT registered as a submodule in the expert; we keep only a callable & out_dim.
#   Private high-freq hash stays inside each expert.

from typing import Optional, Tuple, OrderedDict, Literal
import warnings
import torch
import torch.nn as nn
from torch import Tensor

from models.metamodule import MetaModule, MetaLayerBlock
from models.encodings import SHEncoder, FrequencyEncoder, HashGridEncoder


# -----------------------------
# Utilities
# -----------------------------
def trunc_exp(x: Tensor) -> Tensor:
    # Stable exp with clamp to avoid INF in early steps
    return torch.exp(torch.clamp(x, max=10.0))


# -----------------------------
# Color head (keeps param namespaces stable)
# -----------------------------
class ColorHead(MetaModule):
    """
    Encapsulated color mapping while preserving original param namespaces:
      - 'color_head_hidden'
      - 'color_head_out'
    """

    def __init__(self, in_dim: int, hidden_dim: int, use_sigmoid_rgb: bool) -> None:
        super().__init__()
        self.color_head_hidden = MetaLayerBlock(in_dim, hidden_dim, activation="relu")
        self.color_head_out = MetaLayerBlock(
            hidden_dim, 3, activation="sigmoid" if use_sigmoid_rgb else None
        )

    def forward(self, x: Tensor, *, params: Optional[OrderedDict] = None) -> Tensor:
        h = self.color_head_hidden(
            x, params=self.get_subdict(params, "color_head_hidden")
        )
        rgb = self.color_head_out(h, params=self.get_subdict(params, "color_head_out"))
        return rgb


# -----------------------------
# Base NeRF (shared infra)
# -----------------------------
class MetaNeRFBase(MetaModule):
    """
    Responsibilities:
      - Direction encoding (SH or Fourier) with normalization.
      - Owns color head: concat [geo_feat, enc_dir] → rgb.
      - Public APIs: density(), color(), forward().
      - Children implement: _make_xyz_encoder(), _build_geometry(), _forward_geometry().

    Conventions:
      - density(x) returns dict {"sigma": (...,1), "geo_feat": (...,G)} after activation.
      - forward([xyz|dir]) returns [..., 4] as [rgb(3), sigma(1)] with sigma activated.
    """

    def __init__(
        self,
        *,
        geo_feat_dim: int = 15,
        color_hidden: int = 64,
        use_sigmoid_rgb: bool = True,
        encoder_dir: Literal["spherical", "frequency"] = "spherical",
        include_input_dir: bool = False,  # only applies to "frequency"
    ) -> None:
        super().__init__()
        self.geo_feat_dim = int(geo_feat_dim)
        self.color_hidden = int(color_hidden)
        self.use_sigmoid_rgb = bool(use_sigmoid_rgb)

        # Child defines xyz encoder & geometry trunk
        self.xyz_encoder, in_ch_xyz = self._make_xyz_encoder()
        self._build_geometry(in_ch_xyz=in_ch_xyz, geo_feat_dim=self.geo_feat_dim)

        # Direction encoder
        enc_dir_norm = encoder_dir.lower()
        if enc_dir_norm == "frequency":
            self.dir_encoder = FrequencyEncoder(
                pe_dim=4, include_input=include_input_dir, use_pi=False
            )
            in_ch_dir = self.dir_encoder.out_dim
        elif enc_dir_norm == "spherical":
            self.dir_encoder = SHEncoder(levels=4)
            in_ch_dir = self.dir_encoder.out_dim
            if include_input_dir:
                warnings.warn(
                    "include_input_dir is ignored for spherical encoder.", UserWarning
                )
        else:
            raise ValueError(f"Unsupported encoder_dir: {encoder_dir}")

        # Color head
        self.color_head = ColorHead(
            self.geo_feat_dim + in_ch_dir, self.color_hidden, self.use_sigmoid_rgb
        )

        # Init last layers for stability
        self._init_last_layers()

    # ---------- hooks for children ----------
    def _make_xyz_encoder(self) -> Tuple[nn.Module, int]:
        raise NotImplementedError

    def _build_geometry(self, *, in_ch_xyz: int, geo_feat_dim: int) -> None:
        raise NotImplementedError

    def _forward_geometry(
        self, enc_xyz: Tensor, params: Optional[OrderedDict] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
          sigma_logits: (...,1)  # pre-activation
          geo_feat:     (...,G)
        """
        raise NotImplementedError

    def _init_last_layers(self) -> None:
        """Custom lightweight init for NeRF heads to stabilize early training."""
        with torch.no_grad():
            # ---- Color head last linear ----
            if hasattr(self, "color_head"):
                # The color_head contains a submodule called color_head_out
                ch_out = getattr(self.color_head, "color_head_out", None)
                if ch_out is not None:
                    lin = getattr(ch_out, "linear", None)
                    if lin is not None:
                        lin.weight.mul_(0.1)  # small weights → mild early RGBs
                        if lin.bias is not None:
                            lin.bias.zero_()  # neutral color bias

            # ---- Sigma head ----
            if hasattr(self, "sigma_geo_head"):
                lin = getattr(self.sigma_geo_head, "linear", None)
                if lin is not None:
                    if lin.weight is not None:
                        lin.weight.mul_(0.1)
                    if lin.bias is not None:
                        # First channel = sigma logits bias → negative to avoid density blow-up
                        lin.bias.fill_(0.0)
                        lin.bias[0] = -1.0

    @torch.no_grad()
    def _normalize_dir(self, d: Tensor) -> Tensor:
        return d / (d.norm(dim=-1, keepdim=True) + 1e-9)

    def _encode_dir(self, d: Tensor) -> Tensor:
        d = self._normalize_dir(d)
        return self.dir_encoder(d)

    def _encode_xyz(self, xyz: Tensor) -> Tensor:
        return self.xyz_encoder(xyz)

    # ---------- public API ----------
    def density(self, x: Tensor, *, params: Optional[OrderedDict] = None):
        """
        x: (..., 3) world coords
        returns: (sigma, geo_feat)
            sigma:    (..., 1) with trunc_exp applied
        """
        B = x.shape[:-1]
        enc_xyz = self._encode_xyz(x.view(-1, 3))
        sigma_logits, geo_feat = self._forward_geometry(enc_xyz, params=params)
        sigma = trunc_exp(sigma_logits).view(-1, 1)
        return {"sigma": sigma.view(*B, 1), "geo_feat": geo_feat.view(*B, -1)}

    def color(
        self,
        d: Tensor,  # (...,3) ray directions
        geo_feat: Tensor,  # (...,G)
        *,
        params: Optional[OrderedDict] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        enc_dir = self._encode_dir(d.view(-1, 3))
        rgb = self.color_head(
            torch.cat([geo_feat.view(-1, geo_feat.shape[-1]), enc_dir], dim=-1),
            params=self.get_subdict(params, "color_head"),
        ).view(*geo_feat.shape[:-1], 3)
        if mask is not None:
            out = torch.zeros_like(rgb)
            out[mask] = rgb[mask]
            return out
        return rgb

    def forward(self, x: Tensor, *, params: Optional[OrderedDict] = None) -> Tensor:
        """
        x: (...,6) = [xyz(3), dir(3)]  (world space)
        returns (...,4) = [rgb(3), sigma(1)]  with sigma activated
        """
        assert x.shape[-1] == 6, "Expected input [...,6] = [xyz, dir]"
        B = x.shape[:-1]
        xyz, d = x[..., :3], x[..., 3:]

        enc_xyz = self._encode_xyz(xyz.view(-1, 3))
        sigma_logits, geo_feat = self._forward_geometry(enc_xyz, params=params)
        sigma = trunc_exp(sigma_logits)

        enc_dir = self._encode_dir(d.view(-1, 3))
        rgb = self.color_head(
            torch.cat([geo_feat, enc_dir], dim=-1),
            params=self.get_subdict(params, "color_head"),
        )
        return torch.cat([rgb, sigma], dim=-1).view(*B, 4)

    def extra_repr(self) -> str:
        return f"geo_feat_dim={self.geo_feat_dim}, color_hidden={self.color_hidden}"


# -----------------------------
# Standard NeRF (Fourier xyz)
# -----------------------------
class MetaNeRFStandard(MetaNeRFBase):
    """
    Geometry:
      - xyz encoder: Fourier positional encoding (configurable).
      - MLP with optional skip that re-concats enc_xyz.
      - Head outputs [sigma_logits || geo_feat].
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        depth: int = 4,
        skips: Tuple[int, ...] = (2,),
        pe_dim_xyz: int = 10,
        include_input_xyz: bool = True,
        # shared/base
        geo_feat_dim: int = 15,
        color_hidden: int = 64,
        use_sigmoid_rgb: bool = True,
        encoder_dir: Literal["frequency", "spherical"] = "frequency",
        include_input_dir: bool = False,
    ) -> None:
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.skips = tuple(int(s) for s in skips)
        self.pe_dim_xyz = int(pe_dim_xyz)
        self.include_input_xyz = bool(include_input_xyz)

        super().__init__(
            geo_feat_dim=geo_feat_dim,
            color_hidden=color_hidden,
            use_sigmoid_rgb=use_sigmoid_rgb,
            encoder_dir=encoder_dir,
            include_input_dir=include_input_dir,
        )

    def _make_xyz_encoder(self) -> Tuple[nn.Module, int]:
        enc = FrequencyEncoder(
            pe_dim=self.pe_dim_xyz,
            include_input=self.include_input_xyz,
            use_pi=False,
        )
        return enc, enc.out_dim

    def _build_geometry(self, *, in_ch_xyz: int, geo_feat_dim: int) -> None:
        # Preserve clean param namespaces
        self.backbone_first = MetaLayerBlock(in_ch_xyz, self.hidden, activation="relu")

        layers = []
        in_dim = self.hidden
        for i in range(1, self.depth):
            if i in self.skips:
                in_dim = self.hidden + in_ch_xyz  # skip: concat hidden with enc_xyz
            layer = MetaLayerBlock(in_dim, self.hidden, activation="relu")
            layers.append(layer)
            in_dim = self.hidden
        self.backbone = nn.ModuleList(layers)

        self.sigma_geo_head = MetaLayerBlock(
            self.hidden, 1 + geo_feat_dim, activation=None
        )

    def _forward_geometry(
        self, enc_xyz: Tensor, params: Optional[OrderedDict] = None
    ) -> Tuple[Tensor, Tensor]:
        h = self.backbone_first(
            enc_xyz, params=self.get_subdict(params, "backbone_first")
        )
        for i, layer in enumerate(self.backbone):
            if i + 1 in self.skips:
                h = torch.cat([h, enc_xyz], dim=-1)
            h = layer(h, params=self.get_subdict(params, f"backbone.{i}"))
        sigma_geo = self.sigma_geo_head(
            h, params=self.get_subdict(params, "sigma_geo_head")
        )
        sigma_logits = sigma_geo[..., :1]
        geo_feat = sigma_geo[..., 1:]
        return sigma_logits, geo_feat


# -----------------------------
# Instant-NGP style NeRF (Hash xyz)
# -----------------------------
class MetaNeRFInstant(MetaNeRFBase):
    """
    Instant-NGP style:
      xyz encoder = CONCAT( shared low-freq (optional), private high-freq )
    """

    class _ConcatHashEncoder(nn.Module):
        def __init__(self, shared_call, shared_out_dim, private_enc):
            super().__init__()
            # shared_call: callable or None (not registered)
            self.shared_call = shared_call
            self.shared_out_dim = int(shared_out_dim)
            self.private_enc = private_enc  # registered module

            self.out_dim = self.shared_out_dim + self.private_enc.out_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            parts = []
            if self.shared_call is not None:
                parts.append(self.shared_call(x))
            parts.append(self.private_enc(x))
            return torch.cat(parts, dim=-1)

    def __init__(
        self,
        *,
        aabb: torch.Tensor,  # (2,3)
        hidden: int = 64,
        depth: int = 2,
        # Private high-freq hash config (your encodings.py API):
        high_levels: int = 8,
        high_min_res: int = 64,
        high_max_res: int = 4096,
        high_log2_hashmap_size: int = 17,
        high_features_per_level: int = 2,
        # Optional shared low-freq encoder from container:
        encoder_xyz_shared_low: Optional[HashGridEncoder] = None,
        # base/shared
        geo_feat_dim: int = 15,
        color_hidden: int = 64,
        use_sigmoid_rgb: bool = True,
        encoder_dir: str = "frequency",
        include_input_dir: bool = False,
        **kwargs,
    ) -> None:
        if not (isinstance(aabb, torch.Tensor) and aabb.shape == (2, 3)):
            raise ValueError("aabb must be (2,3) tensor")
        if hidden > 128:
            warnings.warn("Instant-NGP hidden > 128 can slow training.", UserWarning)
        if depth > 4:
            warnings.warn("Instant-NGP depth > 4 is unusual.", UserWarning)

        self.depth = depth
        self.hidden = hidden
        # --- Build encoders FIRST (before super) ---
        # stash AABB (can’t register buffers yet)
        object.__setattr__(self, "_aabb_boot", aabb.detach().to(torch.float32))

        # shared low-freq handle: keep callable + out_dim (do NOT register as submodule)
        shared_call = (
            encoder_xyz_shared_low.__call__
            if encoder_xyz_shared_low is not None
            else None
        )
        shared_out = (
            int(getattr(encoder_xyz_shared_low, "out_dim", 0))
            if encoder_xyz_shared_low is not None
            else 0
        )

        # private high-freq hash (registered module)
        private_enc = HashGridEncoder(
            aabb=self._aabb_boot,
            levels=int(high_levels),
            min_res=int(high_min_res),
            max_res=int(high_max_res),
            log2_hashmap_size=int(high_log2_hashmap_size),
            features_per_level=int(high_features_per_level),
            interpolation="Linear",
        )

        # build a concatenating encoder module we can hand to the base
        concat_enc = self._ConcatHashEncoder(shared_call, shared_out, private_enc)

        # expose these so _make_xyz_encoder can see them
        object.__setattr__(self, "_concat_xyz_encoder", concat_enc)

        # --- now call base init; it will query _make_xyz_encoder() and build geometry ---
        super().__init__(
            geo_feat_dim=geo_feat_dim,
            color_hidden=color_hidden,
            use_sigmoid_rgb=use_sigmoid_rgb,
            encoder_dir=encoder_dir,
            include_input_dir=include_input_dir,
        )

        # finally, register the AABB buffer for nice .to()/.state_dict() behavior
        self.register_buffer("aabb", self._aabb_boot, persistent=True)
        delattr(self, "_aabb_boot")

    # base will call this during __init__
    def _make_xyz_encoder(self) -> Tuple[nn.Module, int]:
        enc = getattr(self, "_concat_xyz_encoder")
        return enc, enc.out_dim

    def _build_geometry(self, *, in_ch_xyz: int, geo_feat_dim: int) -> None:
        self.backbone_first = MetaLayerBlock(in_ch_xyz, self.hidden, activation="relu")
        layers = []
        in_dim = self.hidden
        for _ in range(1, self.depth):
            layers.append(MetaLayerBlock(in_dim, self.hidden, activation="relu"))
            in_dim = self.hidden
        self.backbone = nn.ModuleList(layers)
        self.sigma_geo_head = MetaLayerBlock(
            self.hidden, 1 + geo_feat_dim, activation=None
        )

    def _forward_geometry(
        self, enc_xyz: torch.Tensor, params: Optional[OrderedDict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone_first(
            enc_xyz, params=self.get_subdict(params, "backbone_first")
        )
        for i, layer in enumerate(self.backbone):
            h = layer(h, params=self.get_subdict(params, f"backbone.{i}"))
        sigma_geo = self.sigma_geo_head(
            h, params=self.get_subdict(params, "sigma_geo_head")
        )
        return sigma_geo[..., :1], sigma_geo[..., 1:]
