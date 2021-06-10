import cv2
import torch
import numpy as np
from math import floor


def resize_video(video, size):

    video_len = video.shape[0]
    if video_len == size:
        return video
    frac = video_len / size
    index_map = [ floor(i * frac) for i in range(size)]
    return video[index_map]


def pad_video(video, padding, fill, padding_mode):

    if not isinstance(video, np.ndarray) or len(video.shape) != 4:
        raise TypeError("Got inappropriate video arg")

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")

    if not isinstance(fill, (int, float, tuple, list)):
        raise TypeError("Got inappropriate fill arg")
    elif isinstance(fill, (tuple, list)) and not len(fill) == 3:
        raise TypeError(
            "Got inappropriate fill sequence length {}, expecting 3".format(
                len(fill)))

    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list) and len(padding) != 2:
        raise ValueError("Padding must be an int or a 2 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    if padding_mode not in ["constant", "edge", "wrap"]:
        raise ValueError("Padding mode should be either constant, edge or wrap")

    if isinstance(padding, int):
        pad = (padding, padding)
    else:
        pad = padding

    pad_width = [pad] + [(0, 0)] * 3

    if padding_mode != "constant":
        video = np.pad(video, pad_width, mode=padding_mode)

    elif isinstance(fill, (int, float)):
        constant_values = fill
        video = np.pad(video,
                       pad_width,
                       mode=padding_mode,
                       constant_values=constant_values)

    else:
        frame = np.array(fill).repeat(np.prod(video.shape[1:-1])).reshape(
            video.shape[1:], order=F)
        video = np.stack([
            np.tile(frame, [pad[0], 1, 1, 1]), video,
            np.tile(frame, [pad[1], 1, 1, 1])
        ])

    return video


def rescale_short_side(img, size):
    """Rescales the image such that the short side
        has the specified size

    Adapted from:
    https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/cv2_transform.py

    Arguments:
        img (NumPy Array): A numpy array of shape HxWxC
        size (int): The size to scale the short side of the
            image to
    """
    height, width = img.shape[:-1]
    if (width <= height and width == size) or (height <= width and
                                               height == size):
        return img

    rescaled_width, rescaled_height = size, size

    if width < height:
        # keep the height/width ratio, so scale height according
        # to this
        rescaled_height = floor(height / width * size)
    else:
        # keep the width/height ratio
        rescaled_width = floor(width / height * size)

    # cv2 rescale has the inverse ordering of width, height
    img = cv2.resize(img, (rescaled_width, rescaled_height),
                     interpolation=cv2.INTER_LINEAR)
    return img
