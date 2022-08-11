from collections import OrderedDict

import torch
from torch.utils.data.dataloader import default_collate

from . import BaseDataset


def _flatten(dico, prefix=None):
    """Flatten a nested dictionary."""
    new_dico = OrderedDict()
    if isinstance(dico, dict):
        prefix = prefix + "." if prefix is not None else ""
        for k, v in dico.items():
            if v is None:
                continue
            new_dico.update(_flatten(v, prefix + k))
    elif isinstance(dico, list):
        for i, v in enumerate(dico):
            new_dico.update(_flatten(v, prefix + ".[" + str(i) + "]"))
    else:
        new_dico = OrderedDict({prefix: dico})
    return new_dico


def _unflatten(dico):
    """Unflatten a flattened dictionary into a nested dictionary."""
    new_dico = OrderedDict()
    for full_k, v in dico.items():
        full_k = full_k.split(".")
        node = new_dico
        for k in full_k[:-1]:
            if k.startswith("[") and k.endswith("]"):
                k = int(k[1:-1])
            if k not in node:
                node[k] = OrderedDict()
            node = node[k]
        node[full_k[-1]] = v
    return new_dico


class NestedDictionaryDataset(BaseDataset):

    def __init__(self, defn, sizes=None):
        super().__init__()
        self.defn = _flatten(defn)
        self.sizes = [sizes] if not isinstance(sizes, (list, tuple)) else sizes

        first = None
        for v in self.defn.values():
            if not isinstance(
                    v,
                (
                    BaseDataset,
                    torch.utils.data.Dataset,
                ),
            ):
                raise ValueError("Expected Dataset but found: {}".format(
                    v.__class__))
            first = first or v
            if len(v) > 0:
                assert len(v) == len(first), "dataset lengths must match"

        self._len = len(first)

    def __getitem__(self, index):
        return OrderedDict((k, ds[index]) for k, ds in self.defn.items())

    def __len__(self):
        return self._len

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        for k, ds in self.defn.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except NotImplementedError:
                sample[k] = default_collate([s[k] for s in samples])
        return _unflatten(sample)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(s[index] for s in self.sizes)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if len(self.sizes) == 1:
            return self.sizes[0][index]
        else:
            return (s[index] for s in self.sizes)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return any(ds.supports_prefetch for ds in self.defn.values())

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        for ds in self.defn.values():
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch(indices)

    def set_epoch(self, epoch):
        # TODO(anj): Identify if we need this functionality for evals.
        pass
