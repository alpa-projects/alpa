from examples.opt_serving.dataset import BaseWrapperDataset


class StripTokenDataset(BaseWrapperDataset):

    def __init__(self, dataset, id_to_strip):
        super().__init__(dataset)
        self.id_to_strip = id_to_strip

    def __getitem__(self, index):
        item = self.dataset[index]
        while len(item) > 0 and item[-1] == self.id_to_strip:
            item = item[:-1]
        while len(item) > 0 and item[0] == self.id_to_strip:
            item = item[1:]
        return item
