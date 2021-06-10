import json
import numpy as np
from PIL import Image
import torch


def load_json(json_path):
    """Returns the json dictionary loaded from the specified json path

    Arguments:
        json_path (str): Path to the json file

    Returns:
        json_dictionary: The loaded dictionary corresponding to the json_path
    """
    with open(json_path, 'r') as json_file:
        json_dictionary = json.load(json_file)
    return json_dictionary


def load_image_to_numpy(image_path, mode="RGB"):
    """
    Arguments:
        image_path (str): Path to an image or mask
        mode (str, optional): The mode to convert to

    """
    return np.array(Image.open(image_path).convert(mode))


def load_binary_mask_to_numpy(image_path):
    """
    Arguments:
        image_path (str): Path to the binary mask

    Returns:
        mask (NumPy Array): A numpy array of shape HxW and type uint8
            containing only 0 and 1s
    """
    mask = load_image_to_numpy(image_path, mode="L").astype("uint8")
    mask[mask > 0] = 1
    return mask


def collate_dictionary_fn(data):
    """
    Used as an argument to PyTorch dataloaders in order to output a list of
        dictionaries as the target for each image

    Arguments:
        data (list): A list of length B where data[i] corresponds 
            to the image in the i^th datapoint where data[i][0] is the image
            and data[i][1] is a dictionary containing the target info
    s
    Returns:
        tuple: Tuple (images, batch[1]). Images is a stacked version of all the 
            images in the batch so if each image is C x H x W then the output is
            BxCxHxW. Batch[1] is a list of dictionaries each corresponding to the 
            particular datapoint
    """
    batch = list(zip(*data))
    imgs = torch.stack(batch[0], dim=0)
    return imgs, batch[1]


class VideoCollateFnSelector(object):

    # The video reader might return videos that are corrupted or videos
    # that contains 0 frames. In this case, we allow the users to specify
    # how to deal with it with the collate function, mode details can be found
    # in the __init__ function below

    supprted_modes = {
        "stack_or_combine",
        "remove_and_stack",
        "fill_and_stack",
    }

    def __init__(self, mode, fill=None):
        """
        Used as an argument to PyTorch dataloaders in order to output a list of
            dictionaries as the target for each video

        Arguments:
            mode (str): the mode of this collate function, choose one of the following
                
                stack_or_combine: will take whatever the __getitem__ in the dataset
                returns, and tries to stack them (numpy or torch.Tensor). 
                If stack cannot be performed, either because the clips' shapes are not 
                uniform or because a clip failed to read any frames, it will return a
                list instead. In both cases, the dataloader returns a batch of data 
                with size "batch_size" in the dataloader, but the data can be stacked
                or combined a list depending on their shapes. Corrupted clips will 
                show up as an empty numpy array and target as None

                remove_and_stack: videos that are corrupted will be removed from
                this batch, and the rest of clips will be stacked if the size are
                the same, this will give a good batch, but it is possible that the
                batch returned may be less than "batch size" (If the dataset cannot
                read any frames)

                fill_and_stack: Similar to "remove_and_stack", but in this case, 
                videos that are corrupted will be filled by the "fill"
                argument provided by the user, which must be (frames, target) pairs.
                As a result, the batch size will always be the batch_size specified
                to the dataloader

            fill: (clip_data, target): only required when mode is set to fill_and_stack,
                This will be used to fill the data and targe in case of a corrupted video file. 

        NOTE:
            It is very rare to have video files that contains no frames, and it is also
            dependent on the quality of the video dataset. Therefore, remove_and_stack
            will not still be reflective of the actual batch_size in the dataloader. 

        """

        if mode not in VideoCollateFnSelector.supprted_modes:
            raise ValueError("Mode {} not supported, try {}"\
                .format(mode, ", ".join(VideoCollateFnSelector.supprted_modes)))

        if mode == "fill_and_stack" and fill is None:
            raise ValueError(
                "Fill is required supported when mode = \"fill_and_stack\"")

        if fill is not None and not isinstance(fill, (list, tuple)):
            raise ValueError(
                "Fill must be a tuple of (video, target) pair, got {}".format(
                    fill))

        self.mode = mode
        self.fill = fill

    def __call__(self, data):
        """
        Arguments:
            data (list): A list of length B where data[i] corresponds 
                to the video(frames) in the i^th datapoint where data[i][0] is the video
                and data[i][1] is a dictionary containing the target info
        Returns:
            tuple: Tuple (videos, batch[1]). videos are a stacked version of all the 
                videos in the batch. If each video is T x H x W then the output is
                BxTxCxHxW, otherwise if each video has different Ts the output is a tuple. 
                Batch[1] is a list of dictionaries each corresponding to the particular datapoint

                Also depending on the mode, the clips with 0 frames are either removed, filled
                with something else, or kept un-touched
        """

        batch = zip(*data)
        videos, targets = [list(i) for i in batch]

        if self.mode == "remove_and_stack":
            videos, targets = self.remove_0_frame_videos(videos, targets)

        elif self.mode == "fill_and_stack":
            videos, targets = self.fill_0_frame_videos(videos, targets)

        all_tensors = all([isinstance(x, torch.Tensor) for x in videos])
        all_ndarrays = all([isinstance(x, np.ndarray) for x in videos])

        can_stack = False
        if all_ndarrays or all_tensors:
            shapes = [video.shape for video in videos]
            can_stack = shapes.count(shapes[0]) == len(shapes)

        return self._collate_stack_or_combine(videos, targets, can_stack,
                                              all_tensors, all_ndarrays)

    def remove_0_frame_videos(self, videos, targets):

        ids_to_remove = set()
        for idx in range(len(videos)):
            vid = videos[idx]
            target = targets[idx]
            if len(vid) == 0 or target is None:
                ids_to_remove.add(idx)

        videos = [v for idx, v in enumerate(videos) if idx not in ids_to_remove]

        targets = [
            t for idx, t in enumerate(targets) if idx not in ids_to_remove
        ]

        return videos, targets

    def fill_0_frame_videos(self, videos, targets):

        assert self.fill is not None

        for idx in range(len(videos)):
            vid = videos[idx]
            target = targets[idx]
            if len(vid) == 0 or target is None:
                videos[idx] = self.fill[0]
                targets[idx] = self.fill[1]

        return videos, targets

    def _collate_stack_or_combine(self, videos, targets, can_stack, all_tensors,
                                  all_ndarrays):

        if can_stack:
            if all_tensors:
                videos = torch.stack(videos, dim=0)
            else:
                videos = np.stack(videos, axis=0)

        return videos, targets
