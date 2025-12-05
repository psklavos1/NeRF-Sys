from typing import Dict, List
from torch.utils.data.dataloader import DataLoader


class MultiLoader:
    """
    Grouped, infinite multi-loader for NeRF contexts.

    - Always yields a dict {cid: batch} where cid is the index in `loaders`.
    - Each inner DataLoader is cycled: when it exhausts, we re-create its iterator.
    - Training can run forever; stop via your step budget (outer_steps).
    - Empty loaders are skipped.
    """

    def __init__(self, loaders: List[DataLoader]):
        # keep only non-empty
        self.loaders = [dl for dl in loaders]
        if not self.loaders:
            raise ValueError("MultiLoader received no non-empty DataLoaders.")

        # map each loader to its dataset-provided cell_id
        def get_cid(dl):
            ds = getattr(dl, "dataset", None)
            cid = getattr(ds, "cell_id", None)
            if cid is None:
                raise ValueError("Each dataset must expose .cell_id")
            return int(cid)

        self.cids = [get_cid(dl) for dl in self.loaders]

    def __iter__(self):
        iters = [iter(dl) for dl in self.loaders]
        while True:
            group: Dict[int, dict] = {}
            for i, (dl, it) in enumerate(zip(self.loaders, iters)):
                try:
                    batch = next(it)
                except StopIteration:
                    iters[i] = iter(dl)
                    batch = next(iters[i])
                true_cid = self.cids[i]
                group[true_cid] = batch  # <-- use real cell_id as key
            yield group
