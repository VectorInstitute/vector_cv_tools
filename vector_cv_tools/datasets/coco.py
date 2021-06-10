import torchvision
import numpy as np


def get_name_from_id(class_id, coco_context):
    """
    Arguments:
        class_id (int): The class ID
        coco_context: The initialized cocoapi class

    Returns:
        name (str): The class name corresponding to the ID
    """
    return coco_context.loadCats(class_id)[0]['name']


def remove_empty_kps(coco_context):
    """Returns a list of valid keypoint annotations

    Arguments:
        coco_context: The initialized cocoapi class

    Returns:
        processed_idx (list): List of images with
            valid keypoint annotations
    """
    processed_idx = []
    image_idx_list = coco_context.imgs.keys()
    for idx in image_idx_list:
        ann_ids = coco_context.getAnnIds(imgIds=idx)
        ann_dict = coco_context.loadAnns(ann_ids)
        if len(ann_dict) > 0:
            processed_idx.append(idx)
    return sorted(processed_idx)


def process_coco_keypoints(keypoints):
    """Process a list of keypoints into xy format, since the coco
        keypoints format is (x,y,v). If v is 0(i.e not labelled) then
        it is not included in the return

    Arguments:
        keypoints (list): A flattened list of keypoints where each one is
            in x,y,v format

    Returns:
        processed_idx (list): List of keypoints of the format
            [[x1,y1],[x2,y2], ...]

    """
    processed_keypoints = []
    for i in range(0, len(keypoints), 3):
        if keypoints[i + 2] != 0:
            # reason for adding by 0.5 is b/c
            # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
            processed_keypoints.append(
                [keypoints[i] + 0.5, keypoints[i + 1] + 0.5])
    return processed_keypoints


class CocoDetection(torchvision.datasets.CocoDetection):

    def __init__(self,
                 root,
                 annFile,
                 load_mask=True,
                 load_bbox=True,
                 load_keypoint=False,
                 transforms=None):
        """
        Arguments:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            load_mask (bool, optional): to load the mask from the annotation or not
            load_bbox (bool, optional): to load the bounding box from
                the annotation or not
            load_keypoint (bool, optional): to load the keypoints from
                the annotation or not
            transforms (callable, optional): A callable object that takes in
                img and target as it's input and returns their transformed
                version

        NOTE:
            If load for a feature is set to True (e.g load_mask = True) and the mask
                doesn't exist, then target['masks'] will be an empty list
        """
        super().__init__(root, annFile)
        self._transforms = transforms
        self.load_mask = load_mask
        self.load_bbox = load_bbox
        self.load_keypoint = load_keypoint

    def __getitem__(self, index):
        """
        Arguments:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). Where target is a
                dictionary containing labels, label_names as well as
                masks, boxes, keypoints if their corresponding load is true
        """
        img, anns = super().__getitem__(index)
        img = np.array(img)
        target = {}
        masks, labels, boxes, keypoints = [], [], [], []
        for ann in anns:
            if ann['area'] > 0:
                if self.load_mask:
                    masks.append(self.coco.annToMask(ann).astype("uint8"))
                if self.load_bbox:
                    boxes.append(ann["bbox"])
                if self.load_keypoint and "keypoints" in ann:
                    keypoints.extend(ann['keypoints'])
                labels.append(ann["category_id"])

        if self.load_keypoint:
            keypoints = process_coco_keypoints(keypoints)
        if self.load_mask:
            target["masks"] = masks
        if self.load_bbox:
            target["boxes"] = boxes
        if self.load_keypoint:
            target["keypoints"] = keypoints
        target["labels"] = labels
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # need to get the label_names here because it is entirely possible
        # that the transform drops some labels due to processing
        target['label_names'] = [
            get_name_from_id(label, self.coco) for label in target["labels"]
        ]
        return img, target


class CocoKeypoints(CocoDetection):

    def __init__(self,
                 root,
                 annFile,
                 load_mask=False,
                 load_bbox=False,
                 transforms=None):
        """
        Arguments:
            root (string): Root directory where images are downloaded to.
            annFile (string): Path to json annotation file.
            load_mask (bool, optional): to load the mask from the annotation or not
            load_bbox (bool, optional): to load the bounding box from
                the annotation or not
            transforms (callable, optional): A callable object that takes in
                img and target as it's input and returns their transformed
                version
        """
        super().__init__(root,
                         annFile,
                         load_mask=load_mask,
                         load_bbox=load_bbox,
                         load_keypoint=True,
                         transforms=transforms)
        self.ids = remove_empty_kps(self.coco)
