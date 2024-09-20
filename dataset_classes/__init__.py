from dataset_classes.base_dataset import BaseDataset
from dataset_classes.concat_dataset import ConcatDataset
from dataset_classes.iterable_dataset import IterableDataset
from dataset_classes.packing_dataset import PackingDataset, IterablePackingDataset
from dataset_classes.dna_multimodal_dataset import MultimodalDNADataSet    

from torch.utils.data import DataLoader

class RepeatingLoader:

    def __init__(self, loader: DataLoader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)
        # self.train_token_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # if isinstance(self.loader.dataset, BaseDataset):
            #     self.train_token_count += self.data_iter.dataset.train_token_count
            print("--->Start a new iteration for repeating loader.")
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch
