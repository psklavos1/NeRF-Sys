import os
import sys
from types import SimpleNamespace
from typing import Mapping, Optional, Dict, Any
import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn as nn

from utils import load_checkpoint
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(P: SimpleNamespace, model: torch.nn.Module) -> Optimizer:
    """
    Build an optimizer with parameter groups coming from `model.get_param_groups()`.

    Expected P fields:
        P.optimizer: str in {"adam", "adamw", "sgd"}
        P.lr: float (base LR)
        Optional:
            P.encoding_lr, P.sigma_lr, P.color_lr, P.bg_lr
            P.weight_decay
            P.betas, P.eps  (for Adam/AdamW)
            P.momentum, P.nesterov (for SGD)
            P.alpha          (for RMSprop)
    """
    base_lr: float = getattr(P, "lr", 1e-3)
    weight_decay: float = getattr(P, "weight_decay", 0.0)

    # --------- collect param groups from model ----------
    group_dict: Dict[str, Dict[str, Any]] = model.get_param_groups()
    param_groups = []

    def add_group_if_exists(group_name: str, lr_attr: str):
        if group_name not in group_dict:
            return
        group = group_dict[group_name]
        assert "params" in group, f"{group_name} group must contain 'params'."

        group_lr = getattr(P, lr_attr, None)
        if group_lr is None:
            group_lr = base_lr

        param_groups.append(
            {
                "params": list(group["params"]),
                "lr": float(group_lr),
                "name": group_name,   # <--- crucial for sync_optimizer_lrs
            }
        )

    # These names assume MetaContainer.get_param_groups() returns:
    # {"encoding": {...}, "sigma": {...}, "color": {...}, "background": {...}}
    add_group_if_exists("encoding", "encoding_lr")
    add_group_if_exists("sigma", "sigma_lr")
    add_group_if_exists("color", "color_lr")
    add_group_if_exists("background", "bg_lr")

    # --------- choose optimizer class ----------
    opt_name = getattr(P, "optimizer", "adamw").lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,  # default if group lr missing
            weight_decay=weight_decay,
        )

    elif opt_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
        )

    elif opt_name == "sgd":
        momentum = getattr(P, "momentum", 0.9)
        optimizer = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    return optimizer

def get_scheduler(P: SimpleNamespace, optimizer: Optimizer) -> Optional[_LRScheduler]:
    """
    Optional exponential scheduler that scales all param groups together.

    Expected P fields:
        P.scheduler : str in {"exponential", "none"} (default: "none")
        P.lr        : base initial LR (same used in get_optimizer)
        P.lr_final  : target LR for the *base* group at step P.max_steps
        P.max_steps : total number of training steps (e.g., 20000)
    """
    if P.no_scheduler:
        return None

    max_steps = int(getattr(P, "outer_steps", 0))
    if max_steps <= 0:
        return None

    lr_init = float(getattr(P, "lr", 1e-2))
    lr_final = lr_init / float(getattr(P, "decay_factor", 100))  

    # lr_t = lr_0 * gamma^t  with  lr_max_steps ≈ lr_final
    if lr_final <= 0 or lr_final >= lr_init:
        return None
    gamma = (lr_final / lr_init) ** (1.0 / max_steps)
    return ExponentialLR(optimizer, gamma=gamma)


def verify_param_coverage(model, optimizer):
    grouped = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            grouped.add(id(p))
    all_params = {id(p) for p in model.parameters() if p.requires_grad}
    missing = all_params - grouped
    dup = []
    # optional: detect duplicates by counting
    counts = {}
    for g in optimizer.param_groups:
        for p in g["params"]:
            counts[id(p)] = counts.get(id(p), 0) + 1
    dup = [pid for pid, c in counts.items() if c > 1]

    print(
        f"[OPT VERIFY] total trainable: {len(all_params)}, grouped: {len(grouped)}, missing: {len(missing)}, dups: {len(dup)}"
    )
    if missing:
        names = [n for n, p in model.named_parameters() if id(p) in missing]
        print(
            "[OPT VERIFY] Missing params:", names[:20], "..." if len(names) > 20 else ""
        )
    if dup:
        names = [n for n, p in model.named_parameters() if id(p) in dup]
        print("[OPT VERIFY] Duplicated params:", names)


def print_optimizer_groups(optimizer, title="Current optimizer"):
    print(f"\n===== {title} =====")
    print(f"Total groups: {len(optimizer.param_groups)}")
    for i, g in enumerate(optimizer.param_groups):
        group_name = g.get("name", f"group_{i}")
        param_count = sum(p.numel() for p in g["params"])
        print(
            f"[{i}] name={group_name:<15} | lr={g['lr']:<10.3e} | params={param_count}"
        )
    print("=" * 40)


def print_checkpoint_optimizer_groups(optim_state, title="Loaded optimizer state"):
    groups = optim_state.get("param_groups", [])
    print(f"\n===== {title} =====")
    print(f"Total groups: {len(groups)}")
    for i, g in enumerate(groups):
        group_name = g.get("name", f"group_{i}")
        lr = g.get("lr", None)
        param_refs = g.get("params", [])
        print(
            f"[{i}] name={group_name:<15} | lr={lr:<10.3e} | params={len(param_refs)}"
        )
    print("=" * 40)


def mark_occ_ready_from_state(model: torch.nn.Module, saved_state: dict) -> None:
    """
    If occ grids were present in the saved state, mark .occ_ready = True.
    Otherwise mark False so the pipeline will rebuild or use stratified warmup.
    """
    # Heuristic: look for any '.occ_grid.' key in the loaded state
    has_occ = any(".occ_grid." in k for k in saved_state.keys())

    # Global flag on the container or expert
    if hasattr(model, "occ_ready"):
        model.occ_ready = bool(has_occ)

    # Per-expert optional flags
    if hasattr(model, "submodules"):
        for sub in model.submodules:
            if hasattr(sub, "occ_ready"):
                sub.occ_ready = bool(has_occ)


def is_resume(P, model, optimizer, scheduler=None, scaler=None, prefix="best"):
    """
    Optionally resumes training from a saved checkpoint.

    Returns:
        start_step (int)
        best (float)
        psnr (float)
    """
    start_step = 0
    best = psnr = 0.0
    if P.checkpoint_path is None:
        return start_step, best, psnr

    try:
        # ---- load state dicts ----

        model_state, optim_state, config, scheduler_state, scaler_state = (
            load_checkpoint(
                P.checkpoint_path, mode=prefix)
        )

        # ---- restore model + optimizer ----
        model.load_state_dict(model_state, strict=not getattr(P, "no_strict", False))
        mark_occ_ready_from_state(model=model, saved_state=model_state)

        optimizer.load_state_dict(optim_state)
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        if scaler is not None and scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        # ---- restore step metrics ----
        start_step = config.get("step", 0)
        best = config.get("best", 0.0)
        psnr = config.get("psnr", 0.0)
    except Exception as e:
        print(f"[WARN] Unable to load checkpoint: {e}")

    return start_step, best, psnr


def to_device_tree(x, device):
    """Recursively move tensors in nested containers to device."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, Mapping):
        return {k: to_device_tree(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        ys = [to_device_tree(v, device) for v in x]
        return tuple(ys) if isinstance(x, tuple) else ys
    # leave non-tensors (ints, strings, None, etc.) as-is
    return x


class InfiniteSampler(torch.utils.data.Sampler):
    """
    A sampler that yields an infinite stream of shuffled indices for distributed or standard training.
    Supports shuffling with a limited temporal window to stabilize batches.

    Args:
        dataset (Dataset): Dataset to sample from.
        rank (int): Process rank (for distributed training).
        num_replicas (int): Total number of processes.
        shuffle (bool): Whether to shuffle the data stream.
        seed (int): Random seed for reproducibility.
        window_size (float): Proportion of dataset used as temporal shuffle window.
    """

    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        """
        Iterator that yields indices for the sampler, with optional windowed shuffling.
        Each process gets its own non-overlapping index stream if distributed.
        """
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


class SimpleLogger:
    def __init__(self, filepath: Optional[str]):
        self.filepath = filepath
        self._fh = None
        if filepath:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                self._fh = open(filepath, "w", encoding="utf-8")
            except Exception as e:
                self._fh = None
                print(
                    f"[WARN] Could not open {filepath} for writing: {e}",
                    file=sys.stderr,
                )

    def write(self, msg: str):
        # Always logger.write to stdout
        print(msg, flush=True)
        # Also write to file if available
        if self._fh is not None:
            self._fh.write(msg + ("\n" if not msg.endswith("\n") else ""))
            self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
