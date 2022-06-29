from examples.opt_serving.dataset import BaseWrapperDataset
from examples.opt_serving.dataset import data_utils


class PadDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, left_pad, pad_length=None):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_length = pad_length

    def collater(self, samples):
        return data_utils.collate_tokens(samples,
                                         self.pad_idx,
                                         left_pad=self.left_pad,
                                         pad_to_length=self.pad_length)
