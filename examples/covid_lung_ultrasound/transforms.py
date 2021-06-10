from collections import defaultdict
import cv2
import albumentations as A
from vector_cv_tools import transforms as T
_TRANSFORMS_REGISTRY = {}


# simple struct to represent a video transforms collection
class VideoTransform:

    def __init__(self):
        self.spatial = defaultdict(lambda: None)
        self.temporal = defaultdict(lambda: None)


def register_transform(name, tr_type, split):

    def _decorator(f):

        if name not in _TRANSFORMS_REGISTRY:
            _TRANSFORMS_REGISTRY[name] = VideoTransform()

        transform = _TRANSFORMS_REGISTRY[name]
        if tr_type == "spatial":
            transform.spatial[split] = f()
        elif tr_type == "temporal":
            transform.temporal[split] = f()
        else:
            raise ValueError(
                "Unknown transform type, choose either \"spatial\" or \"temporal\""
            )
        # we really just need the side effect of decorator, so the actual
        # registered function does not matter
        return f

    return _decorator


def get_transform(name):
    return _TRANSFORMS_REGISTRY[name]


def all_transforms():
    return list(_TRANSFORMS_REGISTRY.keys())


@register_transform("base", "spatial", "train")
def base_training_spatial_transforms():
    spatial_transforms_list = [
        A.Resize(80, 80),
        A.ToFloat(max_value=255),
    ]
    return T.ComposeVideoSpatialTransform(spatial_transforms_list)


@register_transform("base", "temporal", "train")
def base_training_temporal_transforms():
    temporal_transforms_list = [
        T.from_albumentation(A.HorizontalFlip()),
        T.RandomTemporalCrop(10),
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)


@register_transform("base", "temporal", "val")
def base_validation_temporal_transforms():
    temporal_transforms_list = [
        T.RandomTemporalCrop(20),
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)


@register_transform("pass_through", "spatial", "train")
def base_training_spatial_transforms_pass_through():
    spatial_transforms_list = [
        A.Resize(112, 112),
        A.ToFloat(max_value=255),
    ]
    return T.ComposeVideoSpatialTransform(spatial_transforms_list)


@register_transform("pass_through", "temporal", "train")
def base_training_temporal_transforms_pass_through():
    temporal_transforms_list = [
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)


@register_transform("pass_through", "spatial", "val")
def base_validation_spatial_transforms_pass_through():
    return base_training_spatial_transforms_pass_through()


@register_transform("pass_through", "temporal", "val")
def base_validation_temporal_transforms():
    temporal_transforms_list = [
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)


@register_transform("fancy", "spatial", "train")
def no_crop_training_spatial_transforms():
    spatial_transforms_list = [
        A.SmallestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(224, 224),
    ]
    return T.ComposeVideoSpatialTransform(spatial_transforms_list)


@register_transform("fancy", "temporal", "train")
def no_crop_training_temporal_transforms():
    temporal_transforms_list = [
        T.TemporalResize(100, mode="pad", padding_mode="wrap"),
        T.from_albumentation(A.CLAHE(always_apply=True)),
        T.from_albumentation(A.HorizontalFlip(p=0.5)),
        T.from_albumentation(A.Rotate(limit=45, p=0.5)),
        T.from_albumentation(A.ColorJitter(p=0.5)),
        T.from_albumentation(A.ToFloat(max_value=255)),
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)


@register_transform("fancy", "spatial", "val")
def no_crop_validation_spatial_transforms():
    spatial_transforms_list = [
        A.SmallestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(224, 224),
        A.ToFloat(max_value=255),
    ]
    return T.ComposeVideoSpatialTransform(spatial_transforms_list)


@register_transform("fancy", "temporal", "val")
def no_crop_validation_temporal_transforms():
    temporal_transforms_list = [
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)


@register_transform("no_crop", "spatial", "train")
def no_crop_training_spatial_transforms():
    spatial_transforms_list = [
        A.SmallestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(224, 224),
        A.ToFloat(max_value=255),
    ]
    return T.ComposeVideoSpatialTransform(spatial_transforms_list)


@register_transform("no_crop", "temporal", "train")
def no_crop_training_temporal_transforms():
    temporal_transforms_list = [
        T.TemporalResize(100, mode="pad", padding_mode="wrap"),
        T.from_albumentation(A.HorizontalFlip(p=0.5)),
        T.from_albumentation(A.Rotate(limit=45, p=0.2)),
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)


@register_transform("no_crop", "spatial", "val")
def no_crop_validation_spatial_transforms():
    spatial_transforms_list = [
        A.SmallestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR),
        A.CenterCrop(224, 224),
        A.ToFloat(max_value=255),
    ]
    return T.ComposeVideoSpatialTransform(spatial_transforms_list)


@register_transform("no_crop", "temporal", "val")
def no_crop_validation_temporal_transforms():
    temporal_transforms_list = [
        T.TemporalResize(100),
        T.ToTensor(),
    ]

    return T.ComposeVideoTemporalTransform(temporal_transforms_list)
