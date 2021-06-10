import matplotlib.pyplot as plt
import cv2
from skimage.color import label2rgb


def plot_multilabel_mask(img,
                         mask,
                         title="",
                         bg_label=0,
                         bg_color=None,
                         save_name="mask"):
    """
    Arguments:
        img (NumPy Array): Image to plot
        mask (NumPy Array): The multilabel mask to overlay
        bg_label (int, optional): Label ID that is treated as background
        bg_color (str/array, optional): The background color specified using
            RGB tuple/color name. Set to None to not have any.
        title (str, optional): Title of the plot
        save_name (str, optional): Name to save the image as, will automatically
            append .png at the end of this name
    """
    to_plot = label2rgb(mask, img.copy(), bg_label=bg_label, bg_color=bg_color)
    plot_image(to_plot, save_name, title)


def plot_binary_mask(img, mask, title="", save_name="binary_mask"):
    """
    Arguments:
        img (NumPy Array): Image to show
        mask (NumPy Array): Binary mask to overlay
        title (str): The title to display
        save_name (str): Name to save the image as
    """
    to_plot = img.copy()
    idx = (mask > 0)
    to_plot[..., 1][idx], to_plot[..., 2][idx] = 1, 1
    plot_image(to_plot, save_name, title)


def plot_image(img, save_name, title=""):
    """A helper function to plot and save the image

    Arguments:
        img (NumPy Array): Image to show
        save_name (str): Name to save the image as, will automatically
            append .png at the end of this name
        title (str, optional): Title of the plot
    """
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.savefig(f"{save_name}.png")
    plt.clf()


def plot_binary_masks(img, masks, label_names):
    """Visualize all the binary masks for a single image

    Arguments:
        img (NumPy Array): NumPy Array of shape C x H x W
        masks (NumPy Array): NumPy Array of shape num_masks x H x W
        label_names (list): A length num_masks list of class labels in string
            form
    """
    for num, (mask, name) in enumerate(zip(masks, label_names)):
        plot_binary_mask(img, mask, name, save_name=f"example_mask_{num}")


def plot_keypoints(img,
                   keypoints,
                   color=(0, 255, 0),
                   diameter=3,
                   save_name="keypoints"):
    """This function is from https://albumentations.ai/docs/examples/example_keypoints/
    """
    img = img.copy()
    for (x, y) in keypoints:
        cv2.circle(img, (int(x), int(y)), diameter, color, -1)
    plot_image(img, save_name)


"""
everything below is code modified from
    https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
"""


def plot_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    """Visualizes a single bounding box on the image
    """
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min +
                                                 w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color,
                  thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name,
                                                     cv2.FONT_HERSHEY_SIMPLEX,
                                                     0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),  #White
        lineType=cv2.LINE_AA,
    )
    return img


def plot_bboxes(img, bboxes, category_names, save_name="bbox"):
    img = img.copy()
    for bbox, class_name in zip(bboxes, category_names):
        img = plot_bbox(img, bbox, class_name)
    plot_image(img, save_name)
