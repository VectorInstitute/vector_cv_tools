import torch
import albumentations as A
from albumentations.core import transforms_interface as albumentations_transforms
from albumentations.core import composition as albumentations_composition
from ..utils import polygon_to_mask


class ComposeAlbumentationsTransform:

    def __init__(self, transforms_list):

        for t in transforms_list:
            if not isinstance(t, (albumentations_transforms.BasicTransform,
                                  albumentations_composition.BaseCompose)):
                raise RuntimeError(
                    "This compose only support albumentations transforms")

        self.transforms_list = transforms_list

    def __call__(self, img):

        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms_list:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ComposeCOCOTransform(ComposeAlbumentationsTransform):

    def __init__(self, transforms_list):
        """
        Arguments:
            transforms_list (list): A list containing albumentations transforms
        """
        super().__init__(transforms_list)
        self.transform = A.Compose(
            transforms_list,
            keypoint_params=A.KeypointParams(format='xy'),
            bbox_params=A.BboxParams(format='coco',
                                     label_fields=['class_labels']))

    def __call__(self, img, target):
        """
        Arguments:
            img (NumPy Array): The input image to transform
            target (dict): The target dictionary to transform

        Returns:
            tuple: Tuple (img, target) where img and target are both transformed

        NOTE:
            This transform will stack keypoints, boxes, masks into torch tensors
            if they exist in the target where:
                target["masks"] is a PyTorch Tensor of shape
                    num_masks x H x W if masks is in target
                target["keypoints"] is a PyTorch Tensor of shape
                    num_keypoints x 2 in (x,y) format if keypoints is in target
                target["boxes"] is a PyTorch Tensor of shape
                    num_boxes x 4 in (x_min, y_min, width, height) format
                    if boxes is in target
        """
        key_list = ["masks", "boxes", "keypoints", "labels"]
        masks, boxes, keypoints, labels = [target.get(x, []) for x in key_list]
        valid_boxes = (len(boxes) > 0)
        labels_to_give = labels if valid_boxes else []
        transformed = self.transform(image=img,
                                     masks=masks,
                                     bboxes=boxes,
                                     class_labels=labels_to_give,
                                     keypoints=keypoints)

        img = transformed["image"]
        keypoints = torch.as_tensor(transformed["keypoints"],
                                    dtype=torch.float32)
        boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
        masks = torch.as_tensor(transformed["masks"], dtype=torch.uint8)
        if valid_boxes:
            labels = transformed['class_labels']
        for key, value in zip(key_list, [masks, boxes, keypoints, labels]):
            if key in target:
                target[key] = value
        return img, target


class ComposeMVTecTransform(ComposeAlbumentationsTransform):

    def __init__(self, transforms_list):
        """
        Arguments:
            transforms_list (list): A list containing albumentations transforms
        """
        super().__init__(transforms_list)
        self.transform = A.Compose(transforms_list)

    def __call__(self, img, target):
        """
        Arguments:
            img (NumPy Array): The input image to transform
            target (dict): The target dictionary to transform

        Returns:
            tuple: Tuple (img, target) where img and target are both transformed
        """
        mask = target["mask"]
        if mask is None:
            transformed = self.transform(image=img)
        else:
            transformed = self.transform(image=img, mask=mask)
            mask = transformed["mask"]
        img = transformed["image"]
        target["mask"] = mask
        return img, target


class ComposeCityscapesTransform(ComposeAlbumentationsTransform):

    def __init__(self, transforms_list, poly_mask_name="masks"):
        """
        Arguments:
            transforms_list (list): A list containing albumentations transforms

        NOTE:
            For polygons, our transform will convert them to masks so that the same
                data augmentation pipeline can be applied to these polygons in mask
                form, this is memory inefficient but ensures that we retain the same
                pipeline that albumentations expects
        """
        super().__init__(transforms_list)
        self.transform = A.Compose(transforms_list)
        # the name to store the resulting masks converted from polygons
        self.poly_mask_name = poly_mask_name

    def __call__(self, img, target):
        """
        Arguments:
            img (NumPy Array): The input image to transform
            target (dict): The target dictionary to transform

        Returns:
            tuple: Tuple (img, target) where img and target are both transformed

        NOTE:
            The returned target dictionary will have the following format if the
                corresponding key exists:

            target["semantic"] is an uint8 PyTorch tensor of shape HxW with pixel wise
                class labels
            target["color"] is an uint8 PyTorch tensor of shape 4xHxW
            target["instance"] is a int32 PyTorch tensor of shape HxW
            target["polygon"] is a dictionary of the following format:
                {
                    "label_ids" : (list: the unique integer IDs for each mask)
                    "label_names" : (list: the names of each mask)
                    "self.poly_mask_name": a PyTorch Tensor of shape num_masksxHxW
                }
        """
        keys = sorted(target.keys())
        if "polygon" in target:
            self.convert_poly_to_mask(target["polygon"])

        # put all the masks in a specified order into a list before the transform
        all_masks, lengths = self.concatenate_masks(keys, target)

        transformed = self.transform(image=img, masks=all_masks)
        img, all_masks = transformed["image"], transformed["masks"]

        # retrieve all the masks in the original order and put them
        # in the appropriate key of the target dictionary
        self.store_masks(all_masks, lengths, keys, target)
        return img, target

    def convert_poly_to_mask(self, polygon_dict):
        """Converts the polygon to masks, and removes the imgHeight, imgWidth
            and polygons fields

        Arguments:
            polygon_dict (dict): A polygon dictionary of the format:

                polygon_dict =
                {
                    "imgHeight" : (int: the height of the image)
                    "imgWidth" : (int: the width of the image)
                    "polygons" : (list: list of polygons)
                    "label_ids" : (list: the unique integer IDs for the
                        polygon objects)
                    "label_names" : (list: the names of the polygon objects)
                }

        Note:
            This function will put the list of converted polygon masks into
                polygon_dict[self.poly_mask_name]
        """
        img_height = polygon_dict.pop('imgHeight')
        img_width = polygon_dict.pop('imgWidth')
        polygons = polygon_dict.pop("polygons")
        polygon_dict[self.poly_mask_name] = [
            polygon_to_mask(poly, img_height, img_width) for poly in polygons
        ]

    def concatenate_masks(self, keys, target):
        """Puts all the masks in a list using the same order as the keys
            passed in and return how much of the list they occupied

        Arguments:
            keys (list): List of keys to iterate over when adding in the
                masks
            target (dict): Dictionary containing the masks

        Returns:
            tuple: Tuple(all_masks, lengths) where all_masks is the list
                of all the masks and lengths is how many masks each key
                has
        """
        all_masks, lengths = [], []
        for key in keys:
            val = target[key] if key != "polygon" else target[key][
                self.poly_mask_name]
            if not isinstance(val, list):
                val = [val]
            lengths.append(len(val))
            all_masks += val
        return all_masks, lengths

    def store_masks(self, all_masks, lengths, keys, target):
        """Recover the transformed masks and put them back into the appropriate key

        Arguments:
            all_masks (list): List of transformed masks
            lengths (list): Lengths of how many masks each key contributed
            keys (list): List of keys to iterate over
            target (dict): Dictionary to store the values into
        """
        start = 0
        for key, len_mask in zip(keys, lengths):
            to_add = torch.as_tensor(all_masks[start:len_mask + start])
            if to_add.size(0) == 1:
                to_add.squeeze_(0)
            if key == "color":
                to_add = to_add.permute(-1, 0, 1)  #HxWxC -> CxHxW
            if key == "polygon":
                target[key][self.poly_mask_name] = to_add
            else:
                target[key] = to_add
            start += len_mask


class ComposeVideoSpatialTransform(ComposeAlbumentationsTransform):

    def __init__(self, transforms_list):
        """
        Arguments:
            transforms_list (list): A list containing albumentations transforms
        """
        super().__init__(transforms_list)
        self.transform = A.Compose(transforms_list)

    def __call__(self, img):
        """
        Arguments:
            img (NumPy Array): The input image to transform
        Returns:
            The transformed image
        """
        return self.transform(image=img)["image"]


class ComposeVideoTemporalTransform:
    """Composes several Video Temporal transforms together.
    """

    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, inputs, target=None):
        for t in self.transforms_list:
            inputs, target = t(inputs, target)
        return inputs, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms_list:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
