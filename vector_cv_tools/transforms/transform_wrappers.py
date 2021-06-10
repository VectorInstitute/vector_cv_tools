import random
from ..utils.misc import np_to_tensor, tensor_to_np


class AlbumentationWrapper:
    """Wraps around an Albumentations spatial_transform so that it can be
        applied to a list of images, where if the transform has something
        random the same augmentation is applied to each image
    """

    def __init__(self, spatial_transform, p=1):
        """
        Arguments:
            spatial_transform: An albumentation transform that can be applied
                to a single image
            p (float, optional): The probability of applying the transform
        """
        self.transform = spatial_transform
        self.p = p

    def __call__(self, list_of_imgs, target=None):
        """To apply a consistent albulmentation transform across all the input
            images, we need to first build an additional_targets dictionary
            and then pass images that have the same keys as the keys
            in additional_targets

        Arguments:
            list_of_imgs: a list of images with length T where each image is of shape
                HxWxC

        Returns:
            transformed version of the video and the target

        NOTE:
            We assume that list_of_imgs must have length >= 1
        """
        if random.random() < self.p:
            additional_targets, to_transform = {}, {}
            for i, value in enumerate(list_of_imgs):
                key = "image" if i == 0 else f"image{i}"
                if i != 0:
                    additional_targets[key] = "image"
                to_transform[key] = value

            # needed by the albulmentations framework
            self.transform.add_targets(additional_targets)

            output = self.transform(**to_transform)

            list_of_imgs = [
                output["image" if i == 0 else f"image{i}"]
                for i in range(len(list_of_imgs))
            ]

        return list_of_imgs, target

    def __repr__(self):
        return self.__class__.__name__ + " for " + str(self.transform)


class TorchvisionWrapper:
    """Wraps around torchvision transform so that it can be
        applied similarily to all elements in a list of images with
        probabilty p.

    It's also possible to use a torchvision functional transform here as long
        as it takes image as it's only input(it's also possible to give a lambda
        function here). As an example we can do something like:
            T.from_torchvision(torchvision.transforms.functional.hflip)

    NOTE:
       ***Do NOT pass in "vision_transform" that contains randomness already
            e.g. torchvision.transforms.RandomCrop ***

        One should only pass in vision_transform that are DETERMINISTIC
            to this wrapper. If you want the crop to be random, but uniformally
            applied to every image in the clip then use the AlbumentationWrapper
                e.g. from_albumentations(A.RandomCrop(224, 224))
    """

    def __init__(self, vision_transform, p=1):
        self.transform = vision_transform
        self.p = p

    def __call__(self, list_of_imgs, target=None):
        """To apply a torchvision transform to our defined data format
            we need to turn the image into CxHxW ordering and then switch
            it back

        Arguments:
            list_of_imgs: a list of images with length T where each image is
                of shape HxWxC

        Returns:
            transformed version of the video and the target
        """
        if random.random() < self.p:
            list_of_imgs = [
                tensor_to_np(self.transform(np_to_tensor(img)))
                for img in list_of_imgs
            ]

        return list_of_imgs, target

    def __repr__(self):
        return self.__class__.__name__ + " for " + str(self.transform)
