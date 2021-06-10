import pickle
import logging
from math import floor
from copy import deepcopy
from hashlib import sha224
from pathlib import Path
from tqdm import tqdm
from ..utils import load_json, VideoReader


def label_format(label):
    """A function that converts the raw label to an appropriate format

    Arguments:
        label (str): The label to format
    """
    return label.lower().replace(" ", "_")


def filter_classes(class_to_id_map, class_filter):
    """
    Arguments:
        class_to_id_map (dict): Dictionary with name to id mapping of classes
        class_filter (list or tuple): List or tuple of classes we want to load
            the segments for

    Returns:
        filtered_dict (dict): Dictionary with name to id mapping of classes
            the classes are the ones in both class_to_id_map and class_filter
    """
    class_filter = [label_format(class_filter) for class_name in class_filter]
    filtered_dict = {
        k: v for (k, v) in class_to_id_map.items() if k in class_filter
    }
    return filtered_dict


def proc_annotation(file_path, ann_dict, split, trimmed, class_to_id_map):
    """Process the annotations and return list of file_paths and corresponding
        annotations

    Arguments:
        file_path (str): File path for the corresponding annotation
        ann_dict (dict): The annotation dictionary
        split (str): train or val or test
        trimmed (bool): To give the trimmed version of the annotation or not
        class_to_id_map (list or tuple): List or tuple of classes we want to
            load the segments for

    Returns:
        filepaths (list): A list of file_paths for this annotation, this is a list
            because in the trimmed setting, multiple segments can be considered
            to be a datapoint
        ann_list (list): A list applicable annotation dictionaries

    NOTES:
        When trimmed is True, it will return anns_list where each element will
            correspond to a seperate valid segment(i.e they are considered
            different datapoints) and then file_path will be duplicated
            so that the video is loaded properly
        When trimmed is False, file_paths will just a single
            element (i.e file_path) and the annotation will contain
            all the segments so anns_list will also be a list with
            1 element.
    """
    ann_list = []
    # check if ann_dict is in the right subset
    if ActivityNetMetadata.SPLIT_TO_SUBSET[split] == ann_dict["subset"]:

        if split == "test":
            return [file_path], [ann_dict]

        anns = ann_dict.pop("annotations", [])
        valid_segments = []
        for ann in anns:
            label = label_format(ann["label"])
            start_s, end_s = ann["segment"]

            if label in class_to_id_map and start_s <= end_s:
                ann["label"] = label
                valid_segments.append(ann)

        if len(valid_segments) > 0:
            tups = [(x["segment"], x["label"], class_to_id_map[x["label"]])
                    for x in valid_segments]
            # if trimmed, consider each segment as a datapoint
            if trimmed:
                for seg, label_name, label_id in tups:
                    new_dict = deepcopy(ann_dict)
                    new_dict["segments"] = [seg]
                    new_dict["label_ids"] = [label_id]
                    new_dict["label_names"] = [label_name]
                    ann_list.append(new_dict)
            else:
                ann_dict["segments"], ann_dict["label_ids"], ann_dict[
                    "label_names"] = map(list, zip(*tups))
                ann_list.append(ann_dict)

    file_paths = [file_path] * len(ann_list)
    return file_paths, ann_list


def filter_anns(filepath_list, anns_list, fps, trimmed):
    """Use the VideoReader to check if the annotation is valid,
        filters out the ones that will not have a frame loaded

    NOTE:
        This function will modify the annotation end_second
            if the end_second exceeds the readable time for the
            trimmed setting
    """
    filtered_paths, filtered_anns = [], []

    for path, ann in tqdm(zip(filepath_list, anns_list),
                          total=len(anns_list),
                          desc="Preprocessing Annotations"):
        cap = VideoReader(path)
        vid_end_s = cap.end_msec / 1000

        if trimmed:
            start_second, end_second = ann["segments"][0]
        else:
            start_second, end_second = 0, vid_end_s

        end_second = min(end_second, vid_end_s)

        if start_second <= end_second:
            source_fps = cap.fps
            min_source_fps = min(source_fps,
                                 fps) if fps is not None else source_fps
            duration = end_second - start_second
            if floor(duration * min_source_fps) > 0:
                if trimmed:
                    ann["segments"][0] = [start_second, end_second]
                filtered_paths.append(path)
                filtered_anns.append(ann)

    return filtered_paths, filtered_anns


def build_anet_metadata(data_path,
                        annotation_path,
                        fps,
                        split,
                        trimmed=True,
                        class_filter=None):
    """Builds and returns the metadata info

    Arguments:
        data_path (Path): The data directory of activitynet in the format of
            <data_path>/<train_val or test>/<videos>
        annotation_path (Path): The path to the activitynet json file.
        split (str): Options are train, test and val.
        class_filter (list or tuple): List or tuple of classes we want to load
            the segments for

    Returns:
        filepath_list (list): A list of all valid video paths
        anns_list (list): A list of all valid annotations
        class_to_id_map (dict): A dictionary containing name -> id mapping of
            valid classes
    """
    ann_dict = load_json(annotation_path)
    class_to_id_map = get_class_to_id_map(ann_dict["taxonomy"])
    data_dict = ann_dict['database']
    filepath_list, anns_list = [], []

    foldername = "train_val" if split in {"train", "val"} else "test"
    get_ann_name = lambda x: x.stem.replace("v_", "")

    if class_filter is not None:
        class_to_id_map = filter_classes(class_to_id_map, class_filter)

    for filename in sorted(data_path.joinpath(foldername).iterdir()):
        annotation_name = get_ann_name(filename)
        filename = str(filename)

        if annotation_name in data_dict:
            filenames, dictionaries_to_add = proc_annotation(
                filename, data_dict[annotation_name], split, trimmed,
                class_to_id_map)

            filepath_list.extend(filenames)
            anns_list.extend(dictionaries_to_add)

    filepath_list, anns_list = filter_anns(filepath_list, anns_list, fps,
                                           trimmed)

    return filepath_list, anns_list, class_to_id_map


def get_class_to_id_map(taxonomy):
    """This function is taken and modified from
        https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/datasets/activitynet.py
        We generate the keys to these dictionaries in with formatted
            by the label format function
    """
    class_names = []
    for node1 in taxonomy:
        is_leaf = True
        for node2 in taxonomy:
            if node2['parentId'] == node1['nodeId']:
                is_leaf = False
                break
        if is_leaf:
            class_names.append(node1['nodeName'])

    class_to_id_map = {}

    for i, class_name in enumerate(sorted(class_names)):
        # offset classes by 1 since class_id 0 should represent
        # the background
        class_to_id_map[label_format(class_name)] = i + 1

    return class_to_id_map


class ActivityNetMetadata:

    SPLIT_TO_SUBSET = {
        "test": "testing",
        "val": "validation",
        "train": "training"
    }

    def __init__(self,
                 data_path,
                 annotation_path,
                 fps=None,
                 split="train",
                 trimmed=True,
                 class_filter=None,
                 use_cache=True):
        """
        data_path (str): The data directory of activitynet in the format of
            <data_path>/<train_val or test>/<videos>
        annotation_path (str): The path to the activitynet json file.
            fps (int, optional): FPS to sample at for the videos, specify
            None to sample at the video's innate FPS. This argument is passed
            into the metadata in order to remove annotations don't have frames
            at the specified FPS
        split (str, optional): Options are train, test and val.
        class_filter (list or tuple, optional): List or tuple of classes we want to load
            the segments for. Specify None to load all classes
        use_cache (bool, optional): whether to use cached metadata information for
            pre-traversed file-structure and dataset information.
        """
        data_path = Path(data_path).resolve()
        annotation_path = Path(annotation_path).resolve()

        if not annotation_path.exists():
            raise ValueError(
                f"Annotation path: {annotation_path} does not exist")

        if not data_path.exists():
            raise ValueError(f"Data path: {data_path} does not exist")

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid Split: {split}")

        self.annotation_path = annotation_path
        self.data_path = data_path
        self.split = split
        self.class_filter = class_filter
        self.trimmed = trimmed
        self.fps = fps

        # if use_cache is false or self.load is false(i.e no preexisting metainfo)
        # then build the metadata and save it
        if not use_cache or not self.load():
            self.paths, self.anns, self.class_to_id_map = build_anet_metadata(
                data_path,
                annotation_path,
                fps,
                split,
                trimmed=self.trimmed,
                class_filter=class_filter)
            self.save()

        if len(self.paths) == 0:
            raise RuntimeError(
                f"data_path: {data_path} has no valid videos with" +
                f" split: {split} and class_filter: {class_filter}")

    @property
    def classes(self):
        """Returns all valid classes and their mappings for the current
            metadata config
        """
        return self.class_to_id_map

    @property
    def cached_data_path(self):
        """Gets the data_path of the current metadata configuration
        """
        hash_info = sha224("{}{}{}{}{}{}".format(
            self.annotation_path, self.data_path, self.split, self.class_filter,
            self.trimmed, self.fps).encode()).hexdigest()
        return Path(f"_anet_md_{hash_info}")

    def get_id_from_name(self, name):
        """Gets the unique integer ID corresponding to the label name.
            Returns None if not found.
        """
        return (self.class_to_id_map[name]
                if name in self.class_to_id_map else None)

    def get_name_from_id(self, idx):
        """Gets the string name corresponding to the idx. Returns None
            if not found.
        """
        for key, class_id in self.class_to_id_map.items():
            if class_id == idx:
                return key
        return None

    def load(self):
        """If cached_data_path exists then load from the file
            and return True. Return False otherwise.
        """
        path = self.cached_data_path
        exists = path.exists()
        if exists:
            with open(path, 'rb') as p_file:
                self.paths, self.anns, self.class_to_id_map = pickle.load(
                    p_file)
        return exists

    def save(self):
        """Save current metadata info to cached_data_path
        """
        path = self.cached_data_path
        if path.exists():
            logging.warning(
                "Metadata file already exists, generating a new one")
        with open(path, 'wb') as p_file:
            pickle.dump((self.paths, self.anns, self.class_to_id_map), p_file)

    def __getitem__(self, index):
        return self.paths[index], self.anns[index]

    def __len__(self):
        return len(self.paths)
