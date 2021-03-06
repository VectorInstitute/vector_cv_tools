import cv2
import numpy as np
import torch
from .video_utils import create_GIF


def visualize_three_views(tensor_list, save_name="view_test"):
    """Visualizes a list of tensors or numpy arrays after the ThreeViews
        transform

    Arguments:
        tensor_list (list): A len 3 list of numpy or tensors of shape TxHxWxC
            where each element contains a different view generated by the
            ThreeViews augmentation
        save_name (str, optional): The name to save, automatically adds the .gif
            extension at the end
    """
    tensor_list = [
        tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
        for tensor in tensor_list
    ]

    create_GIF(f"{save_name}_view1.gif", tensor_list[0])
    create_GIF(f"{save_name}_view2.gif", tensor_list[1])
    create_GIF(f"{save_name}_view3.gif", tensor_list[2])


def visualize_single_segment(seg_array, seg_label, save_name="test"):
    """Visualizes a single segment with it's text label overlayed on top

    Arguments:
        seg_array (Numpy Array): A numpy uint8 array of shape TxHxWxC
        seg_label (str): The text of the label of this segment
        save_name (str, optional): The name to save, automatically adds the .gif
            extension at the end
    """
    for i in range(len(seg_array)):
        to_modify = seg_array[i]
        seg_array[i] = cv2.putText(to_modify.copy(), seg_label, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0),
                                   2)

    create_GIF(f"{save_name}.gif", seg_array)


def visualize_flow_network(flow_array, save_name="flow"):

    flow_vis = []
    for flow in flow_array:

        hsv = np.zeros_like(flow)
        flow = flow.astype(np.float32)
        # convert x,y vector to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # use hue and value for encoding
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        flow_vis.append(rgb)

    create_GIF(f"{save_name}.gif", flow_vis)
