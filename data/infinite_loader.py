import itertools
from torch.utils.data import DataLoader


class InfiniteDataLoader:
    """
    A wrapper that turns any finite DataLoader into an infinite one.

    It will automatically:
    - Restart at the end of each epoch.
    - Reshuffle when the underlying DataLoader uses shuffle=True or a RandomSampler.
    - Keep workers alive if persistent_workers=True.
    - Yield batches forever (no StopIteration).

    Example:
        base_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        loader = InfiniteDataLoader(base_loader)

        for i, batch in enumerate(loader):
            ...
            if i > 1000: break   # user decides when to stop
    """

    def __init__(self, dataloader: DataLoader):
        if not isinstance(dataloader, DataLoader):
            raise TypeError("Expected a torch.utils.data.DataLoader instance.")
        self.dataloader = dataloader
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        # Makes this class usable in `for batch in loader:` loops.
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            # Reset the iterator when the underlying loader is exhausted.
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
        return batch

    # optional convenience for explicit use
    def next(self):
        """Return next batch (alias for `next(loader)`)."""
        return self.__next__()

    def reset(self):
        """Manually reset the underlying iterator (e.g., after changing dataset)."""
        self._iterator = iter(self.dataloader)

    def __len__(self):
        # Length of one epoch (if needed for info)
        return len(self.dataloader)
