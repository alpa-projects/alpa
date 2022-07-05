import numpy as np
import torch

from opt_serving.dataset import BaseWrapperDataset


class NumelDataset(BaseWrapperDataset):

    def __init__(self, dataset, reduce=False):
        super().__init__(dataset)
        self.reduce = reduce

    def __getitem__(self, index):
        item = self.dataset[index]
        if torch.is_tensor(item):
            return torch.numel(item)
        else:
            return np.size(item)

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.reduce:
            return sum(samples)
        else:
            return torch.tensor(samples)
