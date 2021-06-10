import sys
import warnings
import logging
import numpy as np
from torch.utils.data import Dataset
from ..utils import (VideoReader, sample_fraction_extra,
                     validate_fraction_extra)
from .activitynet_metadata import ActivityNetMetadata


class ActivityNet(Dataset):

    def __init__(self,
                 data_path,
                 annotation_path,
                 split="train",
                 fps=None,
                 trimmed=True,
                 max_frames=sys.maxsize,
                 round_source_fps=True,
                 use_cache=True,
                 class_filter=None,
                 fraction_extra=0,
                 spatial_transforms=None,
                 temporal_transforms=None):
        """
        Arguments:
            data_path (str): The data directory of activitynet in the format of
                <data_path>/<train_val or test>/<videos>.
            annotation_path (str): The path to the activitynet json file.
            split (str, optional): Options are train, test and val.
            fps (int, optional): FPS to sample at for the videos, specify
                None to sample at the video's innate FPS.
            trimmed (bool, optional): If set to True then we treat each segment
                as a different datapoint
            max_frames (int, optional): The max amount of frames we will load
                from a given video.
            round_source_fps (bool, optional): If set to True, we round the
                video's innate FPs up to an integer. Useful for videos that
                have FPS 29.97 and need to be rounded
            use_cache (bool, optional): whether to use cached metadata
                information for pre-traversed file-structure and dataset
                information.
            class_filter (list or tuple, optional): List or tuple of classes
                we want to load the segments for. Specify None to
                load all classes

            fraction_extra (int or float or sequence): a number or sequence 
                of length 2 to specify how much extra non action frames we
                want to read in the trimmed setting. If a sequence is
                specified the dataset will uniformally sample a value
                in between the 2 elements inclusively. If a number
                is given then this will be applied determinsitically.
                    For example: if you specify fraction_extra = 0.5
                        then given that the video has enough frames to
                        accommodate this, the clip returned will have
                        50% extra frames relative to the original trimmed
                        setting

            spatial_transforms (callable, optional): Callable transform that
                accepts the image as its argument
            temporal_transforms (callable, optional): Callable transform
                that takes in a Tximage.shape array and a
                target as it's argument

        NOTE:
            Even when trimmed = True, the annotation returned by this class
                will still have keys segments, label_ids, label_names which are
                a list of size 1 in this case to keep the return format consistent
        """
        if trimmed and split == "test":
            raise ValueError(
                "Trimmed cannot be True for split test because no segment " +
                "annotations exist")

        self.metadata = ActivityNetMetadata(data_path,
                                            annotation_path,
                                            fps=fps,
                                            split=split,
                                            trimmed=trimmed,
                                            class_filter=class_filter,
                                            use_cache=use_cache)

        self.split = split
        self.fps = fps
        self.trimmed = trimmed
        self.max_frames = max_frames
        self.round_source_fps = round_source_fps
        self.spatial_transforms = spatial_transforms
        self.temporal_transforms = temporal_transforms

        if not trimmed and fraction_extra != 0:
            warnings.warn("The dataset is not in trimmed mode yet "
                          "fraction_extra has non zero value of "
                          f"{fraction_extra}. "
                          "Overwriting fraction_extra to 0")
            fraction_extra = 0

        validate_fraction_extra(fraction_extra)
        self.fraction_extra = fraction_extra

    @property
    def classes(self):
        """Returns all valid classes and their mappings for the current
            metadata config
        """
        return self.metadata.classes

    @property
    def num_classes(self):
        """Returns the number of classes read by the metadata class
        """
        return len(self.metadata.classes)

    def load_segment(self, vid_iterator, target):
        """Loads all the frames from the vid_iterator, applying spatial
            transform on each frame

        Arguments:
            vid_iterator (iterable): An iterable that returns the frames of the
                video
            target (dict): Dictionary with potentially segments, label_ids
                label_names if an annotation exists for the datapoint. If
                split is test, this dictionary will not have those keys.
                If trim is true then this dictionary will not have the
                segments entry

        Returns:
            Transformed frame_tensor and target
        """

        frames = [
            self.spatial_transforms(img)
            if self.spatial_transforms is not None else img
            for img in vid_iterator
        ]

        num_loaded_frames, num_expected_frames = len(frames), len(vid_iterator)

        if num_loaded_frames == 0:
            logging.warning(
                ("Something went wrong when loading data from clip %s, "
                 "video innate fps: %s, num frames loaded: %s, num frames "
                 "expected: %s"), vid_iterator.path, vid_iterator.fps,
                num_loaded_frames, num_expected_frames)
            frames, target = [], None
        else:
            if self.temporal_transforms is not None:
                frames, target = self.temporal_transforms(frames, target)

        return frames, target

    def sampled_fps(self, cap):
        """Returns the sampled FPS for a given VideoReader instance
        """
        if self.fps is None:
            return cap.rounded_fps if self.round_source_fps else cap.fps
        return self.fps

    def load(self, cap, annotation):
        """Loads the video in full if trimmed is False, otherwise load the
            segment
        """
        frames, target, to_load, = [], None, False
        sampled_frac_extra = 0

        if not self.trimmed or self.split == "test":
            start_second, end_second = 0, None
            to_load = True

        # trimmed annotations are guaranteed to have only
        # one segment
        if self.trimmed and "segments" in annotation:
            start_second, end_second = annotation.pop("segments")[0]
            to_load = True
            sampled_frac_extra = sample_fraction_extra(self.fraction_extra)

        if to_load:
            vid_iterator = cap.to_iter(fps=self.fps,
                                       start_second=start_second,
                                       end_second=end_second,
                                       round_source_fps=self.round_source_fps,
                                       max_frames=self.max_frames,
                                       fraction_extra=sampled_frac_extra)
            frames, target = self.load_segment(vid_iterator, annotation)

        return frames, target

    def __getitem__(self, index):
        vid_path, annotation = self.metadata[index]
        cap = VideoReader(vid_path, load_as_rgb=True)

        data, target = self.load(cap, annotation)

        if target is not None:
            target["sampled_fps"] = self.sampled_fps(cap)
            target["readable_end_second"] = cap.end_msec / 1000
        return data, target

    def __len__(self):
        return len(self.metadata)
