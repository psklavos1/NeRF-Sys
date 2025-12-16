import os
import sys
import argparse
import torch


def build_parser():
    parser = argparse.ArgumentParser(description="TUC's Adaptive NeRF Framework")
    parser.add_argument(
        "--op", type=str, default="train", choices=["train", "eval", "view", "video"]
    )

    # --- system
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")

    # --- io/logging
    parser.add_argument("--eval_step", type=int, default=200)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--print_step", type=int, default=1)
    parser.add_argument(
        "--log_method", type=str, default="step", choices=["step", "patch"]
    )

    # --- data
    parser.add_argument(
        "--dataset",
        type=str,
        default="drz",
        choices=["ffhq", "celeba", "imagenette", "voxceleb", "drz"],
    )
    parser.add_argument(
        "--data_type", type=str, default="ray", choices=["img", "video", "ray"]
    )
    parser.add_argument("--data_path", type=str, default="data/drz/")
    parser.add_argument("--data_dirname", type=str, default="balanced")
    parser.add_argument("--mask_dirname", type=str, default="g22_grid_bm110_ss11")
    parser.add_argument("--cap_images", type=int, default=None)
    parser.add_argument("--downscale", type=float, default=0.25)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--bm", type=float, default=1.05)

    # --- episode gen
    parser.add_argument("--support_rays", type=int, default=4000)
    parser.add_argument("--query_rays", type=int, default=2000)
    parser.add_argument("--cell_dim", type=int, default=5)

    # --- dataloader
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--test_batch_size", type=int, default=1)

    # --- model
    parser.add_argument("--num_submodules", type=int, default=4)
    parser.add_argument(
        "--nerf_variant", type=str, default="instant", choices=["instant", "vanilla"]
    )
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--sigma_depth", type=int, default=2)
    parser.add_argument("--color_depth", type=int, default=2)
    parser.add_argument("--dim_hidden", type=int, default=64)
    parser.add_argument("--color_hidden", type=int, default=64)

    # --- hash encoding
    parser.add_argument("--max_res", type=int, default=4096)
    parser.add_argument("--log2_hashmap_size", type=int, default=20)
    parser.add_argument("--use_occ", action="store_true")
    parser.add_argument(
        "--xyz_encoding", type=str, default="hash", choices=["frequency", "hash"]
    )
    parser.add_argument(
        "--dir_encoding",
        type=str,
        default="spherical",
        choices=["frequency", "spherical"],
    )

    # --- background model
    parser.add_argument("--no_bg_nerf", action="store_true")
    parser.add_argument(
        "--bg_color_default",
        type=str,
        default="random",
        choices=["white", "black", "none", "last_sample", "random"],
    )
    parser.add_argument("--bg_hidden", type=int, default=32)
    parser.add_argument(
        "--bg_encoding",
        type=str,
        default="spherical",
        choices=["frequency", "spherical"],
    )

    # --- rendering
    parser.add_argument("--ray_samples", type=int, default=96)
    parser.add_argument("--chunk_points", type=int, default=262_144 * 17)
    parser.add_argument(
        "--color_space",
        type=str,
        default="linear",
        choices=["srgb", "linear", "identity"],
    )
    # --- FIM
    parser.add_argument("--fim", action="store_true")
    parser.add_argument("--fim_per_sample", action="store_true")
    parser.add_argument("--fim_lambda", type=float, default=0.1)
    parser.add_argument("--fim_beta", type=float, default=0.95)
    parser.add_argument("--fim_epsilon", type=float, default=1e-6)

    # --- optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adamw", "sgd", "adam"]
    )
    parser.add_argument("--encoding_lr", type=float, default=1e-2)
    parser.add_argument("--sigma_lr", type=float, default=2e-3)
    parser.add_argument("--color_lr", type=float, default=2e-3)
    parser.add_argument("--bg_lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)

    # --- scheduler
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument(
        "--decay_factor", type=float, default=10
    )  # final LR = initial LR / decay_factor

    # --- training
    parser.add_argument("--inner_iter", type=int, default=8)
    parser.add_argument("--inner_lr", type=float, default=15e-3)
    parser.add_argument("--outer_steps", type=int, default=20_000)
    parser.add_argument(
        "--algo",
        type=str,
        default="fomaml",
        choices=["maml", "fomaml", "reptile"],
    )
    parser.add_argument("--max_test_tasks", type=int, default=4)

    # --- eval
    parser.add_argument("--tto", type=str, default="16")

    # ----- video & viewing
    parser.add_argument(
        "--camera_path",
        type=str,
        default="full_coverage",
        choices=["spiral_in", "turntable", "east_west", "north_south", "full_coverage"],
    )

    parser.add_argument(
        "--viewer_timeout",
        type=int,
        default=900,  # seconds, e.g. 1 hour=3600; set -1 for "run forever"
        help="Max lifetime for viewer in seconds (-1 = no limit).",
    )
    parser.add_argument("--viewer_public_host", type=str, default="192.168.1.17")

    # --- extras
    parser.add_argument("--configPath", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_date", action="store_true")
    parser.add_argument("--fname", default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--use_stored_args", action="store_true")
    parser.add_argument("--prefix", type=str, default="best")  # best/last/step{i}
    parser.add_argument("--no_strict", action="store_true")

    return parser


def _cli_provided_dests(parser: argparse.ArgumentParser, argv):
    """Return set of 'dest' names explicitly provided on CLI."""
    opt_to_action = {}
    for action in parser._actions:
        for opt in action.option_strings:
            opt_to_action[opt] = action

    provided = set()
    i = 0
    while i < len(argv):
        tok = argv[i]
        act = opt_to_action.get(tok)
        if act is not None:
            provided.add(act.dest)
        i += 1
    return provided


def load_cfg_with_priority(config_path, parser, args, argv):
    """
    Merge a saved config (.P) into args with priority:
    parser defaults < loaded config < explicit CLI
    """
    cli_dests = _cli_provided_dests(parser, argv)

    cfg_obj = torch.load(config_path, map_location="cpu", weights_only=False)
    cfg = vars(cfg_obj) if hasattr(cfg_obj, "__dict__") else dict(cfg_obj)

    for k, v in cfg.items():
        if k not in cli_dests:
            setattr(args, k, v)
    return args


def parse_args():
    """
    Parse CLI args and merge them with checkpoint/external configs; priority is:
    defaults < checkpoint < external config < explicit CLI.
    """
    parser = build_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    cli_dests = _cli_provided_dests(parser, argv)

    if args.checkpoint_path and args.use_stored_args:
        cfg_path = os.path.join(args.checkpoint_path, f"{args.prefix}.P")
        if os.path.exists(cfg_path):
            args = load_cfg_with_priority(cfg_path, parser, args, argv)

    config_path = getattr(args, "configPath", None)
    if config_path is not None:
        import json

        with open(config_path, "r") as f:
            cfg = json.load(f)

        for k, v in cfg.items():
            if not hasattr(args, k):
                continue
            if k in cli_dests:
                continue
            setattr(args, k, v)

    if args.checkpoint_path == "":
        args.checkpoint_path = None

    if args.fname is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.fname = f"{args.op}_{timestamp}"

    return args
