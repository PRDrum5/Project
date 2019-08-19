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
        self.n_test = 0

        samplers = self._train_val_test_sampler(data_split)
        self.train_samp, self.val_samp, self.test_samp = samplers

        super().__init__(dataset=dataset, batch_size=self.batch_size, 
                         sampler=self.train_samp, num_workers=self.n_workers, 
                         shuffle=self.shuffle, drop_last=self.drop_last)
    

    def _train_val_test_sampler(self, data_split):
        train_split, val_split = data_split
        self.n_train = int(round(train_split * self.n_samples))
        self.n_val = int(round(val_split * self.n_samples))
        self.n_test = self.n_samples - self.n_train - self.n_val

        all_idx = np.arange(self.n_samples)

        if self.shuffle:
            np.random.shuffle(all_idx)
        
        val_idx = all_idx[0:self.n_val]

        _n_test = self.n_val + self.n_test
        test_idx = all_idx[self.n_val: _n_test]

        train_idx = all_idx[_n_test: self.n_samples]

        train_sampler = RandomSampler(train_idx)

        samplers = [train_sampler]

        if self.n_val:
            val_sampler = RandomSampler(val_idx)
            samplers.append(val_sampler)

        if self.n_test:
            test_sampler = RandomSampler(test_idx)
            samplers.append(test_sampler)

        self.shuffle = False

        return samplers
    
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

    def train_split(self):
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, 
                          sampler=self.train_samp, num_workers=self.n_workers,
                          drop_last=self.drop_last)

    
    def val_split(self):
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, 
                          sampler=self.val_samp, num_workers=self.n_workers,
                          drop_last=self.drop_last)

    def test_split(self):
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size, 
                          sampler=self.test_samp, num_workers=self.n_workers,
                          drop_last=self.drop_last)
    
    