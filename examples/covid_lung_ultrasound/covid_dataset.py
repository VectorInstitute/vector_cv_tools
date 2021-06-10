import os
import csv
from collections import OrderedDict
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_video
from experiment_utils import read_csv
from vector_cv_tools.utils.misc import np_to_tensor, tensor_to_np
from vector_cv_tools.utils import VideoReader


class COVIDDataset(Dataset):
    # ("other", 3),
    key_to_idx = OrderedDict([
        # we are combining other and normal
        ("normal", 0),
        ("covid", 1),
        ("pneumonia", 2),
    ])

    def __init__(self,
                 csv_file,
                 video_root_dir,
                 labels_to_load=(0, 1, 2, 3),
                 spatial_transforms=None,
                 temporal_transforms=None):
        """Labels to load specifies which of the labels we want to load
            so if its 0, 1 then we only load normal and covid videos

        csv_file is the path to the corresponding split's annotation
        """
        self.video_root_dir = video_root_dir
        csv_tups = read_csv(csv_file)

        # only keep the tups that have labels in labels_to_load
        self.data_tups = [
            (name, label) for name, label in csv_tups if label in labels_to_load
        ]

        self.spatial_transforms = spatial_transforms
        self.temporal_transforms = temporal_transforms

    def __getitem__(self, idx):
        name, label = self.data_tups[idx]
        target = torch.Tensor([label]).long()
        path_to_read = os.path.join(self.video_root_dir, name)

        video_iter = VideoReader(path_to_read).to_iter()
        video_list = [i for i in video_iter]

        if self.spatial_transforms is not None:
            video_list = [
                self.spatial_transforms(frame) for frame in video_list
            ]

        if self.temporal_transforms is not None:
            video_list, target = self.temporal_transforms(video_list, target)

        return video_list, target

    def __len__(self):
        return len(self.data_tups)


class BaselineData(Dataset):

    def __init__(self,
                 input_dims,
                 num=200,
                 spatial_transforms=None,
                 temporal_transforms=None):
        self.num = num

        video = np.random.randint(0, 256, size=input_dims)

        if spatial_transforms is not None:
            video_list = [spatial_transforms(frame) for frame in video]

        target = torch.Tensor([1]).long()

        if temporal_transforms is not None:
            video_list, target = temporal_transforms(video_list, target)

        self.output = video_list
        self.target = target

    def __getitem__(self, idx):
        return self.output, self.target

    def __len__(self):
        return self.num
