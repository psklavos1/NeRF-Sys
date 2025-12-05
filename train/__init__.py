import torch
import torch.nn as nn


def get_inner_lr(P, device, learnable=False):
    """
    Create a learnable inner loop learning rate as a PyTorch parameter.

    Parameters:
    - P: Parameter object or namespace with an 'inner_lr' float attribute.

    Returns:
    - A torch.nn.Parameter representing the learnable inner learning rate.
    """
    return nn.Parameter(
        torch.tensor(P.inner_lr, dtype=torch.float32, device=device),
        requires_grad=learnable,
    )


def setup(algo, P):
    """
    Setup function for meta-learning training.

    This function determines:
    1. The correct training step function based on the algo and data type.
    2. The descriptive filename used to store or identify results.
    3. Whether to include today's date in output tracking.

    Parameters:
    - algo (str): Training algo, e.g., 'maml', 'fomaml', 'non-maml'.
    - P: Parameter object with attributes defining dataset, model config, learning rate, etc.

    Returns:
    - train_step: The training step function appropriate for the current algo and data.
    - fname (str): Formatted name string for checkpoints or logs.
    - today (bool): Whether to include today's date in log naming.
    """
    # Construct default filename string if none was provided
    if P.fname is None:
        if P.data_type == "img":
            fname = (
                f"{P.dataset}/"
                f"{P.data_type}_{P.resolution:03}/"
                f"instep-{P.num_submodules:02}_initer-{P.inner_iter:02}/"
                f"incremental-{P.incremental}_"
                f"lr-{int(P.inner_lr * 1e3):03d}-{int(P.lr * 1e6):04d}_"
                f"single-False"
            )

        elif P.data_type == "video":
            fname = (
                f"{P.dataset}/{P.data_type}_{P.resolution:03}/"
                f"instep-{P.num_submodules:02}_initer-{P.inner_iter:02}/"
                f"incremental-{P.incremental}_"
                f"lr-{int(P.inner_lr * 1e3):03d}-{int(P.lr * 1e6):04d}_"
                f"frozen-{P.frozen}"
            )
        elif P.data_type == "ray":
            data = f"{P.data_type}/{P.dataset}/"
            modularization = f"cells-{P.num_submodules:02}/" + (
                "fim/" if P.fim else "mod/"
            )
            algo_str = f"algo-{P.algo}/"
            model = f"{P.nerf_variant}_dir-{P.dir_encoding}_depth-{P.num_layers}_hid-{P.dim_hidden}_ch-{P.color_hidden}/"
            bg = "no_bg/" if P.no_bg_nerf else f"bg_{P.bg_hidden}/"
            training = f"initer-{P.inner_iter:02}_samples-{P.ray_samples}/"
            optimizer = f"lr-{int(P.inner_lr * 1e3):03d}-{int(P.lr * 1e6):04d}"
            fname = data + modularization + algo_str + model + bg + training + optimizer
    else:
        fname = P.fname

    # Dynamically import the appropriate train_step function
    if algo in ["fomaml", "maml", "reptile"]:
        if P.data_type == "img":
            from train.gradient_based.maml import train_step_img as train_step
        elif P.data_type == "video":
            from train.gradient_based.maml import train_step_video as train_step
        elif P.data_type == "ray":
            from train.gradient_based.nerf_maml import train_step_ray as train_step
    else:
        raise NotImplementedError("Only gradient-based modes are implemented.")

    # Determine whether to use today's date in logging
    today = True if P.log_date else False

    fname += f"_seed-{P.seed}"

    return train_step, fname, today
