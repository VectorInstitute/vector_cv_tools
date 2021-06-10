import torch
import numpy as np
import pickle
import os, sys
from pathlib import Path
import json
import logging
import hashlib
import tqdm
import cv2
from ..utils import video_reader


def flatten_entries(ordered_pairs, data_path, class_filter):

    if class_filter is not None:
        class_filter = set(class_filter)

    mapping = []
    labels = {}
    index = 0
    label_id = 0
    logging.info("generating metadata")
    for video_id, info in tqdm.tqdm(ordered_pairs,
                                    desc="Preprocessing Annotations"):

        annotations = info[0][1]

        LABEL_IDX = 0
        SEGMENT_IDX = 1

        label_name = annotations[LABEL_IDX][1].replace(" ", "_")

        if class_filter is not None and label_name not in class_filter:
            continue

        video_path = data_path / label_name / (video_id + ".mp4")
        if not video_path.exists():
            logging.debug(
                "Path {} cannot be found, skipping".format(video_path))
            continue

        if label_name not in labels:
            labels[label_name] = {"id": label_id, "indexes": []}
            label_id += 1
        labels[label_name]["indexes"].append(index)

        dataset_lookup_entry = {
            "id": video_id,
            "label_id": labels[label_name]["id"],
            "label_name": label_name,
            "path": video_path,
            "segment": annotations[SEGMENT_IDX][1],
        }
        mapping.append(dataset_lookup_entry)

        index += 1

    return mapping, labels


def parse_annotation(annotation_path, data_path, class_filter):

    with open(annotation_path) as f:
        # as opposed to returning a dict object, we return a list
        # since the same video can appear multiple times with different
        # labels and segment start, end times
        raw_data = json.load(f, object_pairs_hook=lambda x: x)
        dataset, labels = flatten_entries(raw_data, data_path, class_filter)

    return dataset, labels


class KineticsMetadata(object):
    """
    Stores metadata information of the entire kinetics 400/600/700 dataset
    """

    def __init__(self,
                 annotation_path,
                 data_path,
                 class_filter=None,
                 target_fps=None,
                 use_cache=True,
                 cache_location=None):
        """
        Initializes a KineticsMetadata object.
        """
        annotation_path = Path(annotation_path).resolve()
        data_path = Path(data_path).resolve()

        if not annotation_path.exists():
            raise RuntimeError(
                "annotation path {} does not exist".format(annotation_path))

        if not data_path.exists():
            raise RuntimeError("data path {} does not exist".format(data_path))

        self.annotation_path = annotation_path
        self.data_path = data_path
        self.class_filter = class_filter
        self.cache_location = cache_location
        self.target_fps = target_fps

        if use_cache:
            if not self.load():
                self.dataset_lookup, self.labels = parse_annotation(
                    annotation_path, data_path, class_filter)
                self.save()
        else:
            self.dataset_lookup, self.labels = parse_annotation(
                annotation_path, data_path, class_filter)

    def cached_data_path(self):
        if self.cache_location is not None:
            loc = self.cache_location

        else:
            loc =  "_kinetics_md_" + \
                hashlib.sha224("{}{}{}".format(
                    self.annotation_path,
                    self.data_path,
                    self.class_filter,
                ).encode()).hexdigest()
        return Path(loc)

    def save(self):
        path = self.cached_data_path()
        if path.exists():
            logging.warning(
                "Metadata file already exists, generating a new one")
        with open(path, 'wb') as f:
            pickle.dump((self.dataset_lookup, self.labels), f)

    def load(self):
        path = self.cached_data_path()
        if not path.exists():
            return False

        with open(path, 'rb') as f:
            self.dataset_lookup, self.labels = pickle.load(f)

        return True

    def __getitem__(self, index):
        return self.dataset_lookup[index]

    def remove(self, index):
        label_name = self.dataset_lookup[index]["label_name"]
        self.labels[label_name]["indexes"].remove(index)
        del self.dataset_lookup[index]

    def __len__(self):
        return len(self.dataset_lookup)


class KineticsDataset(object):
    """
    Dataset for trimmed or untrimmed Kinetics Video dataset
    """

    def __init__(self,
                 annotation_path,
                 data_path,
                 spatial_transforms=None,
                 temporal_transforms=None,
                 fps=None,
                 round_source_fps=True,
                 is_video_trimmed=True,
                 max_frames=None,
                 class_filter=None,
                 use_cache=True,
                 cache_location=None):
        """
        Arguments:
            annotation_path (str): path to the annotation file in json format
            data_path (str): path to the data folder, in the format of
                <data_path>/<label>/<videos>
            spatial_transforms (callable(image), optional): transforms that are applied
                on each frames uniformly, takes in an image and returns an image
            temporal_transforms (callable(frames, target), optional): transforms that are applied
                on each video (TxHxWxC) numpy array and the target, returns a "frames, target" pair
            fps (int, optional): fps value for the output frames, None to use the videos innate FPS
            round_source_fps (bool, optional): If set to True, we round the
                video's innate FPs up to an integer. Useful for videos that
                have FPS 29.97 and need to be rounded
            is_video_trimmed (bool, optional): whether the video data is already trimmed. If so
                will ignore the segment information and only present with action class.
                Otherwise will trim the video and present the action class.
            max_frames (int, optional): maximum number of frames to load for the video clip, the later frames will
                be ignored. Use this to reduce the memory and compute overhead of the data loading,
                note that this is not equivalent to cropping
            class_filter(list or tuple, optional): List or tuple of classes (str) to include only
                a subset of labels and corresponding data, default is None (all).
            use_cache (bool, optional): whether to use cached metadata information for
                pre-traversed file-structure and dataset information.
            cache_location(str, optional): the location of the metadata cache, saved as a pickle
                object. If not provided, will use the hex-digest of annotation path, data path, and
                class filter
        NOTE:
            For spatial transforms that involves randomness (i.e. random crop), be extra
            careful to decide if the same "randomness" should be applied to the entire video dataset
            If so, the transform should be applied as a temporal transform.
        """
        self.metadata = KineticsMetadata(annotation_path,
                                         data_path,
                                         class_filter=class_filter,
                                         target_fps=fps,
                                         use_cache=use_cache,
                                         cache_location=cache_location)

        self.spatial_transforms = spatial_transforms
        self.temporal_transforms = temporal_transforms
        self.fps = fps
        self.round_source_fps = round_source_fps
        self.is_video_trimmed = is_video_trimmed

        self.max_frames = max_frames if max_frames is not None else sys.maxsize

    @property
    def num_classes(self):
        return len(self.metadata.labels)

    def sampled_fps(self, cap):
        """Returns the sampled FPS for a given VideoReader instance
        """
        if self.fps is None:
            return cap.rounded_fps if self.round_source_fps else cap.fps
        return self.fps

    def __getitem__(self, index):
        """
            returns:
                frames(numpy.array), targets(dict)
                frames are TxWxHxC numpy arrays, the T, W, H dimensions
                depend on the specified fps, width, and height, as well as
                any temporal and spatial transforms performed.

                Target is a dictionary containing
                    "label_ids"     : a list of class label IDs,
                    "label_names"   : a list of class label names,
                    "samples_fps"   : the fps this video is sampled at
        """

        datapoint = self.metadata.dataset_lookup[index]
        reader = video_reader.VideoReader(path=str(datapoint["path"]))

        start_sec = 0
        end_sec = None

        # modify the target information based on
        # whether the video is t1rimmed or not
        if not self.is_video_trimmed:
            start_sec, end_sec = datapoint["segment"]

        target = {
            "label_ids": [datapoint["label_id"],],
            "label_names": [datapoint["label_name"],],
            "sampled_fps": self.sampled_fps(reader)
        }

        frames_iter = reader.to_iter(fps=self.fps,
                                     start_second=start_sec,
                                     end_second=end_sec,
                                     round_source_fps=self.round_source_fps,
                                     max_frames=self.max_frames)

        frames = [
            self.spatial_transforms(img)
            if self.spatial_transforms is not None else img
            for img in frames_iter
        ]

        num_loaded_frames, num_expected_frames = len(frames), len(frames_iter)

        if num_loaded_frames == 0:
            # there is a chance that the video is corrupted, we return a
            # empty array and image as empty list, target as None for the
            # collate to handle
            logging.warning(
                ("Something went wrong when loading data from clip %s, "
                 "video innate fps: %s, num frames loaded: %s, num frames "
                 "expected: %s"), datapoint["path"], reader.fps,
                num_loaded_frames, num_expected_frames)
            frames, target = [], None
        else:
            if self.temporal_transforms is not None:
                frames, target = self.temporal_transforms(frames, target)

        return frames, target

    def __len__(self):
        return len(self.metadata)
