from collections.abc import Sequence
import random
import cv2
import torch
import numpy as np


def set_all_randomness(seed, set_for_cuda=True):
    """Sets the random seed for numpy, pytorch, python.random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_for_cuda:
        torch.cuda.manual_seed_all(seed)


def bgr2rgb(frame):
    """Converts a numpy array from BGR ordering of the channels to
        RGB
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def float_to_uint8(img):
    """Converts an numpy array img from range [0-1] to [0-255]
        and casts it to uint8
    """
    uint8_img = img * 255.
    return uint8_img.astype("uint8")


def polygon_to_mask(polygon_list, img_height, img_width):
    """
    Arguments:
        polygon_list (list): A list where each element is of the format (x,y),
            i.e [(x1,y1), (x2,y2) ....]. Note that x,y must be integers
        img_height (int): Height of the image we want to make the mask for
        img_width (int): Width of the image we want to make the mask for

    Returns:
        A binary mask of uint8 type with shape HxW
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    pts = np.array([polygon_list], dtype=np.int32)
    cv2.fillPoly(mask, pts, 1)
    return mask


def np_to_tensor(img):
    """Converts a numpy image of shape HxWxC to PyTorch tensor of shape
        CxHxW
    """
    return torch.from_numpy(img).permute(-1, 0, 1)


def tensor_to_np(tensor):
    """Converts a PyTorch tensor of shape CxHxW back to a numpy array
        of shape HxWxC
    """
    return tensor.permute(1, 2, 0).numpy()


def validate_fraction_extra(fraction_extra):
    """Validates fraction_extra to ensure that it's either an int or float
        or a Sequence or length 2
    """
    if not isinstance(fraction_extra, (int, float, Sequence)):
        raise TypeError("fraction_extra must be a int or float or sequence."
                        " Got {}".format(type(fraction_extra)))

    if isinstance(fraction_extra, Sequence) and len(fraction_extra) != 2:
        raise ValueError("If fraction_extra is a sequence it "
                         "should have length of 2. Got length of {}".format(
                             len(fraction_extra)))


def sample_fraction_extra(fraction_extra):
    """If fraction_extra is a not a number then sample a value
        uniformally between the 2 elements, otherwise return
        fraction_extra
    """
    if not isinstance(fraction_extra, (int, float)):
        fraction_extra = random.uniform(fraction_extra[0], fraction_extra[1])

    return fraction_extra
