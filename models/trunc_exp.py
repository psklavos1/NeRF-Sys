import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

# dtype-dependent safe max log for exp(x)
_EXP_MAX = {
    torch.float16: 11.089866488,  # ~log(65504)
    torch.bfloat16: 88.722839111,  # ~log(3.4e38)
    torch.float32: 88.722839111,
    torch.float64: 709.782712893,
}


def _exp_clamp(x: torch.Tensor) -> torch.Tensor:
    m = _EXP_MAX.get(x.dtype, _EXP_MAX[torch.float32])
    # symmetric clamp; lower bound is just for numerical sanity
    return x.clamp(-m, m)


class _TruncExpFn(Function):
    @staticmethod
    @custom_fwd()  # keep caller dtype under autocast; we handle clamp per dtype
    def forward(ctx, x):
        xc = _exp_clamp(x)
        y = torch.exp(xc)
        ctx.save_for_backward(xc)  # save *clamped* input for consistent grad
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        (xc,) = ctx.saved_tensors
        # dy/dx = exp(xc) — same clamp as forward
        return grad_out * torch.exp(xc)


def trunc_exp(x: torch.Tensor) -> torch.Tensor:
    return _TruncExpFn.apply(x)
