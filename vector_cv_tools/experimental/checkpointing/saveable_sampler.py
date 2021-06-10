import torch
from torch.utils.data.sampler import Sampler
from .base_interface import SaveableInterface, validate_dict


class SaveableSampler(Sampler, SaveableInterface):
    """With a very large training data, an epoch may take longer than your permitted
        time slice. Therefore, you might want to save checkpoint after certain
        iterations within an epoch. However, if you are randomizing the batches
        for each epoch, you need to save the state of the sampler in case of
        preemption. PyTorch currently does not have the out-of-box solution,
        but it's possible to use this sampler class to do so
    """

    def __init__(self, dataset, shuffle=False):
        self.data = dataset
        self.shuffle = shuffle

        # initial dataloader index
        self.init_index()

    def init_index(self):
        if self.shuffle:
            self.indices = torch.randperm(len(self.data))
        else:
            self.indices = torch.arange(len(self.data))

        self.data_counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.data_counter == len(self.data):
            self.init_index()
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter]
            self.data_counter += 1
            return int(ele)

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = (
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) *
                    batch_size)
        return {
            'indices': self.indices,
            'data_counter': self.data_counter - prefetched_num,
        }

    def load_state_dict(self, state_dict):
        validate_dict(state_dict, "indices", "data_counter")
        self.indices = state_dict['indices']
        self.data_counter = state_dict['data_counter']
