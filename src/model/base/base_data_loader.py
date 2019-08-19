import numpy as np
from torch.utils.data import DataLoader

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, 
                 n_workers=1, drop_last=True):

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.drop_last = drop_last

        self.n_samples = len(dataset)

        super().__init__(dataset=dataset, batch_size=self.batch_size, 
                         num_workers=self.n_workers, shuffle=self.shuffle, drop_last=self.drop_last)
    