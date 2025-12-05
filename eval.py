import torch
from torch.utils.data import DataLoader

from common.args import parse_args
from common.utils import InfiniteSampler, load_model_checkpoint
from data.dataset import get_dataset
from main import collate_fn
from models.model import get_model
from utils import Logger, set_random_seed
from argparse import ArgumentParser
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--filename", type=str, default="best")
    parser.add_argument("--tto", help="Test time optimization", default="1", type=str)
    args = parser.parse_args()
    return args


def main():
    """argument define"""
    args = parse_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ load P """
    P_path = os.path.join(args.load_path, f"{args.filename}.P")
    P = torch.load(P_path)

    """ define logger """
    logger = Logger(P.fname, ask=False, today=False, rank=P.rank)
    logger.log(f"\nargs: {args}")
    logger.log(f"\nP: {P}")

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset """
    train_set, test_set = get_dataset(P, dataset=P.dataset)

    """ define dataloader """
    kwargs = {"pin_memory": True, "num_workers": 0}
    train_sampler = InfiniteSampler(
        train_set, rank=0, num_replicas=1, shuffle=True, seed=P.seed
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=P.batch_size,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs
    )

    """ Initialize model """
    model = get_model(P).to(device)
    model_path = os.path.join(args.load_path, f"{args.filename}.model")
    model_ckpt = torch.load(model_path)
    not_loaded = model.load_state_dict(model_ckpt, strict=not bool(P.no_strict))
    print(f"not_loaded: {not_loaded}")

    """ define train and test type """
    from evals import setup as test_setup

    test_func = test_setup(P.algo, P)

    """ test """
    ttos = [int(tto) for tto in args.tto.split(",")]
    for step in ttos:
        P.tto = step
        set_random_seed(P.seed)
        test_func(
            P, model, test_loader, step, logger=logger, prefix=f"{args.filename}_tto"
        )


if __name__ == "__main__":
    main()
