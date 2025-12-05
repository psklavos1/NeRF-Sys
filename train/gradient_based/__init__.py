from collections import OrderedDict
import math
import random
import torch
from torch import optim
from einops import rearrange, repeat
from copy import deepcopy
from utils import rsvrBase
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inner_adapt(
    P,
    wrapper,
    task_data,
    inner_lr=1e-2,
    num_steps=4,
    first_order=False,
    params=None,
    order=None,
):
    """
    Perform the inner-loop adaptation using gradient-based updates.

    Args:
        P: Configuration object.
        wrapper: MetaWrapper object encapsulating the task-specific model.
        task_data: Support set for one task.
        inner_lr: Inner-loop learning rate.
        num_steps: Number of gradient steps.
        first_order: If True, disables higher-order gradient computation.
        params: Initial parameters (optional).
        order: Loss tiling strategy ('raster', 'rowwise', 'colwise').

    Returns:
        Updated parameters, raw losses, divided losses, reconstructions, and gradients.
    """
    loss = 0.0
    batch_size = (
        int(task_data[0]["rays"].shape[0])
        if P.data_type == "ray"
        else (
            int(task_data["videos"].size(0)) if P.data_type in ["video", "img"] else -1
        )
    )

    params = wrapper.get_batch_params(params, batch_size)  # as many as episodes!
    losses_in, grads_in = [], []
    loss_in_log, res_in_log = [], []

    num_iters = P.inner_iter if wrapper.training else P.tto

    """ inner gradient step """
    for step_inner in range(num_steps):
        for step_iter in range(num_iters):
            if P.data_type == "img":
                # Perform an inner update step on image data- performed per patch
                params, loss_in, loss_in_div, res_in, grad_in = inner_loop_step_img(
                    P,
                    wrapper,
                    params,
                    task_data,
                    inner_lr,
                    first_order,
                    step_inner,
                    step_iter,
                )
            elif P.data_type == "video":
                # Perform an inner update step on video data
                params, loss_in, loss_in_div, res_in, grad_in = inner_loop_step_video(
                    P,
                    wrapper,
                    params,
                    task_data,
                    inner_lr,
                    first_order,
                    step_inner,
                    step_iter,
                )
            elif P.data_type == "ray":
                pass
            else:
                raise NotImplementedError()

            losses_in.append(loss_in)
            grads_in.append(grad_in)

            if step_iter == num_iters - 1:
                loss_in_log.append(loss_in_div)
                res_in_log.append(res_in)

    return (
        params,
        torch.stack(losses_in),
        torch.stack(loss_in_log),
        res_in_log,
        grads_in,
    )


def inner_loop_step_img(
    P,
    wrapper,
    params,
    task_data,
    inner_lr=1e-2,
    first_order=False,
    step_inner=-1,
    step_iter=-1,
):
    """
    One step of the inner loop for image data.

    Returns:
        Updated parameters, loss, tiled loss, reconstruction, and gradient.
    """
    num_steps = P.num_submodules
    is_incremental = P.incremental

    b, c, h, w = task_data.size()
    wrapper.decoder.zero_grad()

    with torch.enable_grad():
        loss_in, res_in = wrapper(
            task_data, params=params, step_inner=step_inner, step_iter=step_iter
        )
        loss_in_div = divide_loss(loss_in, num_steps, P.order)
        # for i, loss in enumerate(loss_in_div):
        #     print(f"Division {i} loss sum: {loss.sum().item():.6f}")
        if is_incremental:  # If incremental training, divide loss for each step
            loss_grad = loss_in_div[step_inner].view(b, -1).mean(dim=-1)
        else:
            loss_grad = loss_in.view(b, -1).mean(
                dim=-1
            )  # per sample scalal loss (b, cxhxw)

        grads = torch.autograd.grad(
            loss_grad.mean() * b,
            params.values(),
            create_graph=not first_order,
            allow_unused=True,
        )
        
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            grad = grad if grad is not None else 0.0
            if P.oml and f"layers.{P.oml_layer}.linear" not in name:
                updated_params[name] = param
            else:
                updated_params[name] = param - inner_lr * grad

    return updated_params, loss_in, loss_in_div, res_in, grads



def inner_loop_step_video(
    P,
    wrapper,
    params,
    task_data,
    inner_lr=1e-2,
    first_order=False,
    step_inner=-1,
    step_iter=-1,
):
    """
    One step of the inner loop for video data.

    Returns:
        Updated parameters, loss, tiled loss, reconstruction, and gradient.
    """
    num_steps = P.num_submodules
    is_incremental = P.incremental
    b, t, c, h, w = task_data.size()
    wrapper.decoder.zero_grad()

    if P.prog:
        # Handle progressive network growth
        truncated_params = OrderedDict()
        for name, param in params.items():
            if "bias" in name or "weight" in name:
                feats_new = (
                    P.dim_hidden // P.num_submodules * (step_inner + 1)
                    + P.dim_hidden % P.num_submodules
                )
                feats_prev = (
                    P.dim_hidden // P.num_submodules * step_inner
                    + P.dim_hidden % P.num_submodules
                )
                decoder_params = wrapper.get_batch_params(None, P.batch_size)[name]

                if f"layers.{P.oml_layer}.linear" in name:
                    param_new = (
                        decoder_params[..., :feats_new]
                        if "bias" in name
                        else decoder_params[..., :feats_new, : P.dim_hidden]
                    )
                    if step_iter == 0:
                        if "bias" in name:
                            param_new[..., :feats_prev] = param[..., :feats_prev]
                        else:
                            param_new[..., :feats_prev, : P.dim_hidden] = param[
                                ..., :feats_prev, : P.dim_hidden
                            ]
                    truncated_params[name] = param_new
                elif f"layers.{P.oml_layer + 1}.linear" in name and "weight" in name:
                    param_new = decoder_params[..., : P.dim_hidden, :feats_new]
                    if step_iter == 0:
                        param_new[..., : P.dim_hidden, :feats_prev] = param[
                            ..., : P.dim_hidden, :feats_prev
                        ]
                    truncated_params[name] = param_new
                else:
                    truncated_params[name] = param
            else:
                truncated_params[name] = param
    else:
        truncated_params = params

    with torch.enable_grad():
        loss_in, res_in = wrapper(
            task_data,
            params=truncated_params,
            step_inner=step_inner,
            step_iter=step_iter,
        )
        loss_in_div = rearrange(loss_in, "b t c h w -> t b c h w")

        loss_grad = (
            loss_in_div[step_inner].view(b, -1).mean(dim=-1)
            if is_incremental
            else loss_in.view(b, -1).mean(dim=-1)
        )

        grads = torch.autograd.grad(
            loss_grad.mean(),
            truncated_params.values(),
            create_graph=not first_order,
            allow_unused=True,
        )

        updated_params = OrderedDict()
        for (name, param), grad in zip(truncated_params.items(), grads):
            grad = grad if grad is not None else 0.0

            # Frozen gradient masking for specific OML layers
            if (
                f"layers.{P.oml_layer}.linear" in name
                or f"layers.{P.oml_layer+1}.linear" in name
            ):
                if P.frozen:
                    width = (
                        P.dim_hidden // P.num_submodules * step_inner
                        + P.dim_hidden % P.num_submodules
                    )
                else:
                    width = 0

                if "weight" in name:
                    if f"{P.oml_layer}.linear" in name:
                        grad[..., :width, :] = 0
                    else:
                        grad[..., :, :width] = 0
                elif "bias" in name:
                    grad[..., :width] = 0

            updated_params[name] = (
                param - inner_lr * grad
                if f"layers.{P.oml_layer}.linear" in name
                else param
            )

    return updated_params, loss_in, loss_in_div, res_in, grads


def divide_loss(loss, num_steps, order):
    b, c, h, w = loss.size()
    if order == "raster":
        n = int(math.isqrt(num_steps))
        assert n * n == num_steps
        hs = torch.linspace(0, h, n + 1).round().long().tolist()
        ws = torch.linspace(0, w, n + 1).round().long().tolist()
        tiles = [
            loss[..., hs[i] : hs[i + 1], ws[j] : ws[j + 1]]
            for i in range(n)
            for j in range(n)
        ]
        return torch.stack(tiles)
    elif order == "rowwise":
        return torch.stack(torch.tensor_split(loss, num_steps, dim=-2))
    elif order == "colwise":
        return torch.stack(torch.tensor_split(loss, num_steps, dim=-1))
    else:
        raise NotImplementedError
