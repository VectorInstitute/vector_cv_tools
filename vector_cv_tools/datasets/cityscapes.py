import torchvision
import numpy as np


def process_polygon_dictionary(polygon_dict, load_license_plates=False):
    """Processes and mutates the polygon_dict, this function will convert
        and remove the objects field to result in the following format after
        execution:

            polygon_dict =
            {
                "imgHeight" : (int: the height of the image)
                "imgWidth" : (int: the width of the image)
                "polygons" : (list: list of polygons)
                "label_ids" : (list: the unique integer IDs for the
                    polygon objects)
                "label_names" : (list: the names of the polygon objects)
            }

    Arguments:
        polygon_dict (dict): A dictionary containing the polygons loaded
            from Cityscapes dataset
        load_license_plates (bool, optional): To load the license plate class
            or not
    """
    to_process = polygon_dict.pop("objects")

    list_of_tups = [(d["label"], Cityscapes.get_id_from_name(d["label"]),
                     d["polygon"])
                    for d in to_process
                    if load_license_plates or d["label"] != "license plate"]

    label_names, label_ids, polygons = map(list, zip(*list_of_tups))

    polygon_dict["polygons"] = polygons
    polygon_dict["label_ids"] = label_ids
    polygon_dict["label_names"] = label_names


class Cityscapes(torchvision.datasets.Cityscapes):

    def __init__(self,
                 root,
                 split="train",
                 target_type="semantic",
                 load_license_plates=False,
                 transforms=None):
        """Supports only the "fine" quality mode of Cityscapes

        Arguments:
            root (string): Root directory of the Cityscapes Dataset
            split (string, optional): Options are train, test, val
            target_type (string or list, optional): Can be instance, semantic,
                polygon or color. Specify a list of these to load them
                in that order
            load_license_plates (bool, optional): To load the license plate class
                or not
            transforms (callable, optional): A callable object that takes in
                img and target as it's input and returns their transformed
                version
        """
        super().__init__(root,
                         split=split,
                         mode="fine",
                         target_type=target_type)
        self._transforms = transforms
        self.load_license_plates = load_license_plates

    @staticmethod
    def get_id_from_name(name):
        """A static method that gets the unique integer class id from the
            class name

        Arguments:
            name (str): The class name

        Returns:
            The unique integer id corresponding to the class name
                if found, otherwise return None
        """
        for named_tup in Cityscapes.classes:
            if named_tup.name == name:
                return named_tup.id
        return None

    @staticmethod
    def get_name_from_id(idx):
        """A static method that gets name of the class given its integer
            class id

        Arguments:
            idx (int): The unique integer ID to find the name for

        Returns:
            The name of the class if found, otherwise return None
        """
        for named_tup in Cityscapes.classes:
            if named_tup.id == idx:
                return named_tup.name
        return None

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). Where target is a dictionary with
                an entry for each of the specified target_types where:

            target[type] = mask for types that are not polygon
            target["polygon"] =
            {
                "imgHeight" : (int: the height of the image)
                "imgWidth" : (int: the width of the image)
                "polygons" : (list: list of polygons)
                "label_ids" : (list: the unique integer IDs for the
                    polygon objects)
                "label_names" : (list: the names of the polygon objects)
            }
        """
        img, target_tup = super().__getitem__(index)
        img, target = np.array(img), {}

        if not isinstance(target_tup, tuple):
            target_tup = (target_tup,)

        for key, value in zip(self.target_type, target_tup):
            if key != "polygon":
                value = np.array(value)
            else:
                process_polygon_dictionary(value, self.load_license_plates)
            target[key] = value

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
