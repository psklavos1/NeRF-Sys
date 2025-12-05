import functools
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.args import parse_args
from common.utils import (
    InfiniteSampler,
    MultiLoader,
    get_optimizer,
    load_model_checkpoint,
)
from data.dataset import get_dataset
from models.model import get_model
from train import get_inner_lr
from train.trainer import meta_trainer
from utils import Logger, set_random_seed


def collate_fn(batch):
    len_batch = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if len_batch > len(batch):
        diff = len_batch - len(batch)
        for i in range(diff):
            batch = batch + batch[:diff]
    return torch.utils.data.dataloader.default_collate(batch)


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id
    split_size = len(dataset.video_paths) // worker_info.num_workers
    dataset.video_paths = dataset.video_paths[
        worker_id * split_size : (worker_id + 1) * split_size
    ]


def main(rank, P):
    P.rank = rank
    P.incremental = True  # ? Debug

    """ set torch device"""
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    """ fixing randomness """
    set_random_seed(P.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    """ define dataset """
    train_set, test_set = get_dataset(P, dataset=P.dataset)

    train_loader_kwargs = {
        "batch_size": P.batch_size,
        "collate_fn": collate_fn,
        "num_workers": 8,
        "prefetch_factor": 2,
    }
    test_loader_kwargs = {
        "batch_size": P.test_batch_size,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": 0,
    }
    """ define dataloader """
    if P.data_type == "ray" and isinstance(train_set, list):
        # one DataLoader per subgrig
        # TODO: Consider Inifinite sampler
        train_loaders = [
            DataLoader(ds, **train_loader_kwargs, shuffle=True) for ds in train_set
        ]
        test_loaders = [DataLoader(ds, **test_loader_kwargs) for ds in test_set]
        train_loader = MultiLoader(train_loaders)
        test_loader = MultiLoader(test_loaders)
    else:
        train_sampler = InfiniteSampler(
            train_set, rank=rank, num_replicas=1, shuffle=True, seed=P.seed
        )
        train_loader = DataLoader(
            train_set, sampler=train_sampler, **train_loader_kwargs
        )
        test_loader = DataLoader(test_set, **test_loader_kwargs)

    """ Initialize model, optimizer """
    inner_lr = get_inner_lr(P)
    model = get_model(P).to(device)
    optimizer = get_optimizer(P, model, inner_lr)

    """ define train and test type """
    from evals import setup as test_setup
    from train import setup as train_setup

    train_func, fname, today = train_setup(P.algo, P)
    P.fname = fname
    test_func = test_setup(P.algo, P)

    """ define logger """
    logger = Logger(fname, ask=P.checkpoint_path is None, today=today, rank=P.rank)
    logger.log(P)
    logger.log(model)

    """ load model if necessary """
    load_model_checkpoint(P, model, logger)

    """ apply data parrallel for multi-gpu training """
    if P.data_parallel:
        raise NotImplementedError()  # Currently having some error with DP

    input("only pipeline reamining")
    """ train """
    meta_trainer(
        P,
        train_func,
        test_func,
        model,
        optimizer,
        train_loader,
        test_loader,
        logger,
        inner_lr,
    )

    """ close tensorboard """
    logger.close_writer()


if __name__ == "__main__":
    """argument define"""
    P = parse_args()

    P.world_size = torch.cuda.device_count()
    P.data_parallel = P.world_size > 1

    main(0, P)
