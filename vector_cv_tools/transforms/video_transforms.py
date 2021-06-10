"""
Every temporal transform in vector_cv_tools will need to take in
a list with length >= 1 of numpy images of shape HxWxC and target as it's
callable arguments even if target is not modified.
"""
import warnings
import torch
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from ..utils.misc import np_to_tensor, tensor_to_np
from . import functional as F
from .transform_wrappers import AlbumentationWrapper, TorchvisionWrapper


class ToTensor:
    """A ToTensor transform that conforms to Video Dataset's specs

    NOTE:
        This transform assumes that each image in the list_of_imgs of images
            has the same shape
    """

    def __call__(self, list_of_imgs, targets=None):
        if isinstance(list_of_imgs, list):
            list_of_imgs = np.stack(list_of_imgs, axis=0)

        return torch.from_numpy(list_of_imgs), targets

    def __repr__(self):
        return self.__class__.__name__ + '()'

class TemporalResize(torch.nn.Module):
    """Resize the video in temporal dimension to the desired size
    If the image is torch Tensor or numpy array, it is expected
    to have [..., T, H, W] shape, where ... means an arbitrary number of leading dimensions,
    Args:
        size (int): Desired output size of the resize.
        mode (str): what do when "size" is larger than the original size of the
            video. Should be one of: repeat or pad, default is repeat
             - repeat: repeat the intermediary frames
             - pad: pad the video based on padding mode
        padding_mode (str): Type of padding. Should be one of: constant, edge,
            reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the first/last frame of the video
             - wrap: pads with repetition of the video
        padding_fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively (as 0-255).
            This value is only used when the padding_mode is constant.
    """

    def __init__(self,
                 size,
                 mode="repeat",
                 padding_mode="constant",
                 padding_fill=0):

        super().__init__()

        if not isinstance(size, int):
            raise ValueError("Please provide a single int for size")

        if mode not in ("repeat", "pad"):
            raise ValueError("Mode must be one of repeat or pad")

        self.size = size
        self.mode = mode
        self.padding_mode = padding_mode
        self.padding_fill = padding_fill

    def forward(self, list_of_imgs, target=None):
        """
        Arguments:
            list_of_imgs: a list of images with length T where each image is
                of shape HxWxC

        Returns:
            transformed list_of_imgs and target
        """
        video_len = len(list_of_imgs)
        if video_len == self.size:
            return list_of_imgs, target

        video = np.array(list_of_imgs)
        if self.mode == "repeat" or video_len > self.size:
            video = F.resize_video(video, self.size)
        else:
            padding_needed = self.size - video_len
            # pad on both ends
            padding = [
                padding_needed // 2, padding_needed // 2 + padding_needed % 2
            ]
            video = F.pad_video(video, padding,
                                self.padding_fill, self.padding_mode)

        return list(video), target

    def apply_to_target(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)

class RandomTemporalCrop(torch.nn.Module):
    """Crop the given video at a random location in time.
    If the image is torch Tensor or numpy array, it is expected
    to have [..., T, H, W] shape, where ... means an arbitrary number of leading dimensions,
    Args:
        size (int): Desired output size of the crop.
        padding (int or sequence): Optional padding on either end
            of the video. Default is None. If a single int is provided, it is
            used pad both beginning and end of the video, if a sequence of length 2
            is provided, it is used to pad start and end of the video, respectively
        pad_if_needed (boolean): It will pad the video if shorter than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively (as 0-255).
            This value is only used when the padding_mode is constant.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the first/last frame of the video
             - wrap: pads with repetition of the video
    """

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=True,
                 fill=0,
                 padding_mode="constant"):

        super().__init__()

        if not isinstance(size, int):
            raise RuntimeError("Please provide a single int for size")

        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, list_of_imgs, target=None):
        """
        Arguments:
            list_of_imgs: a list of images with length T where each image is
                of shape HxWxC

        Returns:
            transformed list_of_imgs and target
        """
        video = np.array(list_of_imgs)

        # this is the padding defined by the user
        if self.padding is not None:
            video = F.pad_video(video, self.padding, self.fill, self.padding_mode)

        length = video.shape[0]

        start = 0
        # pad the width if needed
        if self.pad_if_needed and length <= self.size:
            padding_needed = self.size - length
            padding = [
                padding_needed // 2, padding_needed // 2 + padding_needed % 2
            ]
            video = F.pad_video(video, padding, self.fill, self.padding_mode)

        else:
            start = np.random.randint(length - self.size)

        return list(video[start:start + self.size]), target

    def apply_to_target(self):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)


class RandomSpatialCrop(torchvision.transforms.RandomCrop):
    """Applies the torchvision RandomCrop transform but on a list of images
        where each image will be cropped with the same coordinates in the list
        of images

    This class is taken and adapted from:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomCrop

    The init paramters of this class has the same meaning as the link above
    """

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=True,
                 fill=0,
                 padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def _pad_image(self, img):
        """Pads a torch image of shape HxWxC
        """
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        return img

    def forward(self, list_of_imgs, target=None):
        """
        Arguments:
            list_of_imgs: a list of images with length T where each image is
                of shape HxWxC

        Returns:
            transformed version of the video and the target
        """
        # pad the first image to compute the parameter for the crop
        first_padded_img = self._pad_image(np_to_tensor(list_of_imgs[0]))

        # get the randomized parameters and fix it
        i, j, h, w = self.get_params(first_padded_img, self.size)

        output_image_list = []
        for num, img in enumerate(list_of_imgs):
            # first image is already padded
            img = first_padded_img if num == 0 else self._pad_image(
                np_to_tensor(img))

            img = tensor_to_np(TF.crop(img, i, j, h, w))

            output_image_list.append(img)

        return output_image_list, target


class RandomResizedSpatialCrop(torchvision.transforms.RandomResizedCrop):
    """Applies the torchvision RandomSizedCrop transform but on a list of images
        where each image will be have the exact same randomness applied to it

    The init parameters has the same functionality as:
    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)

    def forward(self, list_of_imgs, target=None):
        first_img = np_to_tensor(list_of_imgs[0])
        i, j, h, w = self.get_params(first_img, self.scale, self.ratio)

        output_image_list = []
        for num, img in enumerate(list_of_imgs):
            img = first_img if num == 0 else np_to_tensor(img)

            img = tensor_to_np(
                TF.resized_crop(img, i, j, h, w, self.size, self.interpolation))

            output_image_list.append(img)

        return output_image_list, target


class SampleEveryNthFrame:
    """Samples the list of images at every Nth frame, note that the 0th
        frame is always included
    """

    def __init__(self, n):
        """
        Arguments:
            n (int): the frame interval to sample the list_of_imgs at
        """
        self.n = n

    def __call__(self, list_of_imgs, target=None):
        list_of_imgs = [
            list_of_imgs[i] for i in range(0, len(list_of_imgs), self.n)
        ]
        return list_of_imgs, target

    def __repr__(self):
        return self.__class__.__name__ + f"(n={self.n})"


class RandomShortSideScaleJitter:

    def __init__(self, min_size, max_size):
        """
        Arguments:
            min_size (int): The min to sample and scale the short side of the
                image to
            max_size (int): The max to sample and scale the short side of the
                image to
        """
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, list_of_imgs, target=None):

        size = round(np.random.uniform(self.min_size, self.max_size))

        list_of_imgs = [F.rescale_short_side(img, size) for img in list_of_imgs]

        return list_of_imgs, target

    def __repr__(self):
        return (self.__class__.__name__ +
                f"(min={self.min_size}, max={self.max_size})")


class RescaleShortSide:

    def __init__(self, size):
        """
        Arguments:
            size (int): The size to scale the short side of the image to
        """
        self.size = size

    def __call__(self, list_of_imgs, target=None):

        list_of_imgs = [
            F.rescale_short_side(img, self.size) for img in list_of_imgs
        ]

        return list_of_imgs, target

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.size})"


class ThreeViews:
    """
    Taken and adapted from:
    https://github.com/open-mmlab/mmaction2/blob/3a3e10a8e3c92a0dd73be60a43f99b07eb6181a4/mmaction/datasets/pipelines/augmentations.py#L1521

    Crops the image into 3 equally sized crops

    NOTES:
        Normally you will want to apply this transform after
            RescaleShortSide

    IMPORTANT:
        THIS MUST BE THE TERMINAL TRANSFORM(i.e SHOULD NOT APPLY ANY OTHER
            TRANSFORM, not even ToTensor)
    """

    def __init__(self, crop_H, crop_W, stack_as_tensor=True):
        """

        Arguments:
            crop_H (int): The height to scale to
            crop_W (int): The width to scale to
            stack_as_tensor (bool): Stack as a Tensor if True np array if False
        """
        self.crop_H, self.crop_W = crop_H, crop_W
        self.stack_as_tensor = stack_as_tensor

    def __call__(self, list_of_imgs, target=None):
        """
        Returns the target unmodified and a list of video clips with the
            following format:
                [first_view_clip, second_view_clip, third_view_clip]
        """
        H, W = list_of_imgs[0].shape[:-1]
        crop_H, crop_W = self.crop_H, self.crop_W

        if crop_H == H:
            w_step = (W - crop_W) // 2
            offsets = [
                (0, 0),  # left
                (w_step, 0),  # middle
                (2 * w_step, 0),  # right
            ]
        elif crop_W == W:
            h_step = (H - crop_H) // 2
            offsets = [
                (0, 0),  # top
                (0, h_step),  # middle
                (0, 2 * h_step),  # down
            ]
        else:
            warnings.warn(f"{self} is not performed because at least one" +
                          " side must be equal to the crop size")
            return list_of_imgs, target

        cropped = []
        for img in list_of_imgs:
            crop = [
                img[y_offset:y_offset + crop_H, x_offset:x_offset + crop_W]
                for x_offset, y_offset in offsets
            ]
            cropped.append(crop)

        map_fn = torch.as_tensor if self.stack_as_tensor else np.array

        # a list of [first_view_clip, second_view_clip, third_view_clip]
        clips = list(map(map_fn, zip(*cropped)))

        return clips, target

    def __repr__(self):
        return "{}(crop_H={}, crop_W={}, stack_as_tensor={}".format(
            self.__class__.__name__, self.crop_H, self.crop_W,
            self.stack_as_tensor)


def from_albumentation(spatial_transform, p=1):
    """Takes in a spatial albumentations transform and wrap it such that
        it can be applied to a list of images where each image in that list
        will be applied with the exact same transform even if the
        spatial_transform contains randomness.

    Returns a callable class that takes in list of images and target as input

    If you can't find a transform on a video that is implemented, it is
        very likely that Albumentation has the something similar for images
        that you can use, for example it has RandomSizedCrop that can be used
        as well. But we've implemented one that makes the paramteres match
        the torchvision counterpart
    """
    return AlbumentationWrapper(spatial_transform, p=p)


def from_torchvision(vision_transform, p=1):
    """Takes in an arbitary torchvision tranform and wrap it such that it can be
        applied to a list of images of shape HxWxC

    Returns a callable class that takes in list of images and target as input

    NOTE:
        Due to implementation difficuities, in order to apply the same
            randomized transform to EACH image, it is best to pass in
            a deterministic transform like the functional transforms
            in torchvision and then pass in a p value for the wrapper
            to roll a number and apply the transform with that probability

        Additionally, it's also possible to wrap a torchvision functional transform
            as long as it's a function that takes in an image as it's only argument
            i.e can write something like:
                lambda x: some_functional_transform(x,...)
    """
    return TorchvisionWrapper(vision_transform, p=p)
