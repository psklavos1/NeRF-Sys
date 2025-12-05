import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class ImgTokenizer(nn.Module):
    """
    Converts input images into a sequence of patch embeddings with positional encodings.

    Args:
        input_size (int or tuple): Spatial size of the input image (H, W).
        patch_size (int or tuple): Size of square patches to extract (pH, pW).
        dim (int): Output embedding dimension.
        padding (int or tuple): Optional padding before unfolding patches.
        img_channels (int): Number of input channels (e.g., 3 for RGB).
    """

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=3):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.patch_size = patch_size
        self.padding = padding

        # Linear projection from flattened patch to embedding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * img_channels, dim)

        # Compute number of patches and initialize positional embeddings
        n_patches = ((input_size[0] + padding[0] * 2) // patch_size[0]) * (
            (input_size[1] + padding[1] * 2) // patch_size[1]
        )
        self.posemb = nn.Parameter(torch.randn(n_patches, dim))

    def forward(self, x):
        # Extract non-overlapping patches and flatten them
        x = F.unfold(
            x, self.patch_size, stride=self.patch_size, padding=self.padding
        )  # (B, C*p*p, L)
        x = x.permute(0, 2, 1).contiguous()  # (B, L, C*p*p)

        # Project to embeddings and add positional encodings
        x = self.prefc(x) + self.posemb.unsqueeze(0)  # (B, L, dim)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        dim (int): Input feature dimension.
        n_head (int): Number of attention heads.
        head_dim (int): Dimension per head.
        dropout (float): Dropout rate for attention output.
    """

    def __init__(self, dim, n_head, head_dim, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim**-0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None):
        if to is None:
            to = fr  # Self-attention if no 'to' input provided

        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)

        # Reshape for multi-head computation
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_head),
            [q, k, v],
        )

        # Attention score computation
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)  # (B, H, N, N)

        # Apply attention weights
        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = einops.rearrange(out, "b h n d -> b n (h d)")  # (B, N, dim)

        return self.to_out(out)


class FeedForward(nn.Module):
    """
    Standard MLP with GELU activation used in transformer blocks.

    Args:
        dim (int): Input/output feature dimension.
        ff_dim (int): Hidden dimension for intermediate linear layer.
        dropout (float): Dropout rate.
    """

    def __init__(self, dim, ff_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    """
    Pre-normalization wrapper for any layer (typically attention or feedforward).

    Args:
        dim (int): Input dimension.
        fn (nn.Module): Function to wrap (e.g., attention or FFN).
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class TransformerEncoder(nn.Module):
    """
    A transformer encoder stack composed of multiple layers of attention + feedforward blocks.

    Args:
        dim (int): Input/output feature dimension.
        depth (int): Number of transformer layers.
        n_head (int): Number of attention heads.
        head_dim (int): Dimension per attention head.
        ff_dim (int): Dimension of feedforward subnetwork.
        dropout (float): Dropout rate.
    """

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for norm_attn, norm_ff in self.layers:
            x = x + norm_attn(x)  # Residual connection + Attention
            x = x + norm_ff(x)  # Residual connection + FeedForward
        return x
