import os
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import load_image_to_numpy, load_binary_mask_to_numpy

MVTec_OBJECTS = ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
                 'transistor', 'wood', 'zipper')  #15 different objects

MVTec_CON_TYPES = (
    "good", "bent", "bent_lead", "bent_wire", "broken", "broken_large",
    "broken_small", "broken_teeth", "cable_swap", "color", "combined",
    "contamination", "crack", "cut", "cut_inner_insulation", "cut_lead",
    "cut_outer_insulation", "damaged_case", "defective", "fabric_border",
    "fabric_interior", "faulty_imprint", "flip", "fold", "glue", "glue_strip",
    "gray_stroke", "hole", "liquid", "manipulated_front", "metal_contamination",
    "misplaced", "missing_cable", "missing_wire", "oil", "pill_type", "poke",
    "poke_insulation", "print", "rough", "scratch", "scratch_head",
    "scratch_neck", "split_teeth", "squeeze", "squeezed_teeth", "thread",
    "thread_side", "thread_top")  # 49 contamination types


def validate_arguments(split, obj_types, con_types):
    """Checks the arguments to MVTec dataset

    Arguments:
        split (str): String indicating train or test split
        obj_types (list): List containing strings indicating
            the different objects to load
        con_types (list): List containing strings indicating
            the different contaminations to load

    """
    if split not in {"train", "test"}:
        raise ValueError(f"Split {split} is not supported")
    for obj in obj_types:
        if obj not in MVTec_OBJECTS:
            raise ValueError(f"Invalid object {obj}")
    for con in con_types:
        if con not in MVTec_CON_TYPES:
            raise ValueError(f"Invalid contamination type {con}")


def id_to_object(obj_id):
    """
    Arguments:
        obj_id (int): Object's numerical ID

    Returns:
        String that corresponds to the object's numerical ID
    """
    return MVTec_OBJECTS[obj_id]


def object_to_id(obj_name):
    """
    Arguments:
        obj_name (str): The object name

    Returns:
        ID that corresponds to the the object's name
    """
    return MVTec_OBJECTS.index(obj_name)


def id_to_contaimination(con_id):
    """
    Arguments:
        con_id (int): Contamination's numerical ID

    Returns:
        String that corresponds to the contamination's numerical ID
    """
    return MVTec_CON_TYPES[con_id]


def contamination_to_id(con_name):
    """
    Arguments:
         con_name (str): The contamination name

    Returns:
        ID that corresponds to the the contamination's name
    """
    return MVTec_CON_TYPES.index(con_name)


def get_valid_paths(root, folder_type, obj_types, contamination_types):
    """Gets all valid image paths depending on the specified
        object types and contamination_types

    Arguments:
        root (str): The root path of the MVTec dataset directory
        folder_type (str): This can be either train, test or ground_truth
        obj_types (list): List of strings containing the object types to load
        contamination_types (list): List of strings containing the
            contamination_types to load

    Returns:
        all_paths (list): A sorted list of valid image paths
    """
    all_paths, contamination_types = [], set(contamination_types)
    for obj in set(obj_types):
        file_pattern = os.path.join(root, f"{obj}/{folder_type}/*/*.png")
        for potential_path in glob(file_pattern):
            _, label = get_label_from_path(potential_path)
            if label in contamination_types:
                all_paths.append(potential_path)
    return sorted(all_paths)


def get_mask_path(path_to_img):
    """Gets the path to mask from path to image

    e.g. if path_to_img is:
        .../object_type/folder_type/broken_large/000.suffix
        then the path to mask will be
        .../object_type/ground_truth/broken_large/000_mask.suffix

    Arguments:
        path_to_img (str): Path to the image

    Returns:
        mask_path (str): Path to the corresponding mask,
            note that this path might not actually exist
    """
    p = Path(path_to_img)
    mask_name = f"{p.stem}_mask{p.suffix}"
    mask_path = list(p.parts)
    mask_path[-1] = mask_name
    mask_path[-3] = "ground_truth"
    mask_path = os.path.join(*mask_path)
    return mask_path


def get_label_from_path(path_to_img):
    """From a path to image, get a tuple of the format
        (obj_name, contamination_type) e.g (bottle, broken_large)

    Arguments:
        path_to_img (str): Path to the image
    """
    p = Path(path_to_img)
    obj_type = os.path.basename(p.parents[2]).lower()
    contamination_type = os.path.basename(p.parent).lower()
    return obj_type, contamination_type


class MVTec(Dataset):

    def __init__(self,
                 root,
                 split="train",
                 obj_types=None,
                 con_types=None,
                 mask_only=False,
                 transforms=None):
        """
        Arguments:
            root (string): Root directory of the MVTec dataset
            split (string): Options are train and test
            obj_types (list, optional): List of strings containing the object
                types to load
            con_types (list, optional): List of strings containing the contaminations
                to load
            mask_only (bool, optional): If the split is test, this decides if 
                we only load from images that have a corresponding mask(i.e
                containmation that is not of type "good")
            transforms (callable, optional): A callable object that takes in
                img and target as it's input and returns their transformed
                version

        NOTE:
            The train set for MVTec has no masks, so leave mask_only to be False and
                con_types to be None or ["good"] to load images in the train set
        """
        obj_types = [x.lower() for x in obj_types
                    ] if obj_types is not None else MVTec_OBJECTS
        con_types = [x.lower() for x in con_types
                    ] if con_types is not None else MVTec_CON_TYPES
        # removing good type contaminations will ensure that only images with mask
        # will get loaded
        if mask_only:
            con_types = [x for x in con_types if x != "good"]
        validate_arguments(split, obj_types, con_types)
        paths = get_valid_paths(root, split, obj_types, con_types)
        if len(paths) == 0:
            raise ValueError(
                "No data points available for the specified combination")
        self.paths = paths
        self.load_mask = (split == "test")
        self._transforms = transforms

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        Returns:
            tuple: Tuple (img, target). Where target is a dictionary containing
            the label names and label ids in the format:
            {
                "label_names" : ("the type of object", "the type of contamination"),
                "label_ids" : (int: the unique integer ID of the object,
                    int: the unique integer ID of the contamination )
            }

        NOTE:
            Normal/good images have contamination type and ID as "good". and 0,
                respectively.
            If the mask doesn't exist for the image, target['mask'] will be None
        """
        path_to_load, mask = self.paths[index], None
        img = load_image_to_numpy(path_to_load, mode="RGB")
        obj_type, con_type = get_label_from_path(path_to_load)
        mask_path = get_mask_path(path_to_load)
        if self.load_mask and os.path.exists(mask_path):
            mask = load_binary_mask_to_numpy(mask_path)
        target = {}
        target["label_names"] = (obj_type, con_type)
        target["label_ids"] = (object_to_id(obj_type),
                               contamination_to_id(con_type))
        target["mask"] = mask
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.paths)
