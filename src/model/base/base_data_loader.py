import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, 
                 data_split, n_workers=1, drop_last=True):

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.drop_last = drop_last

        self.n_samples = len(dataset)
        self.n_train = 0
        self.n_val = 0

        self.train_samp, self.val_samp = self._train_val_sampler(data_split)

        super().__init__(dataset=dataset, batch_size=self.batch_size, 
                         sampler=self.train_samp, num_workers=self.n_workers, 
                         shuffle=self.shuffle, drop_last=self.drop_last)
    
    
    def _train_val_sampler(self, data_split):
        #TODO Improve this to include test. Include in config
        self.n_train = int(round(data_split * self.n_samples))
        self.n_val = self.n_samples - self.n_train

        all_idx = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(all_idx)
        
        if self.n_val > 0:
            val_idx = all_idx[0:self.n_val]
            train_idx = np.delete(all_idx, np.arange(0, self.n_val))

            train_sampler = RandomSampler(train_idx)
            val_sampler = RandomSampler(val_idx)
            samplers = (train_sampler, val_sampler)
        else:
            sampler = RandomSampler(all_idx)
            samplers = (sampler, None)

        self.shuffle = False

        return samplers
    
    def val_split(self):
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, 
                          sampler=self.val_samp, num_workers=self.n_workers,
                          drop_last=self.drop_last)
    