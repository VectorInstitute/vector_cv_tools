import torch.nn as nn
from torchvision.models.video import r3d_18, r2plus1d_18, mc3_18

_MODEL_REGISTRY = {}


def register_model(cls):
    _MODEL_REGISTRY[cls.__name__] = cls
    return cls


def all_models():
    return list(_MODEL_REGISTRY.keys())


def get_model(name):
    return _MODEL_REGISTRY[name]


@register_model
class R3D18(nn.Module):

    def __init__(self,
                 num_classes=3,
                 pretrained=False,
                 freeze_backbone=False,
                 dropout_prob=0):
        super().__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        model = r3d_18(pretrained=pretrained, progress=False)

        for param in model.parameters():
            param.requires_grad = not freeze_backbone
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


@register_model
class Res2P1Model(nn.Module):

    def __init__(self,
                 num_classes=3,
                 pretrained=False,
                 freeze_backbone=False,
                 dropout_prob=0):
        super().__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        model = r2plus1d_18(pretrained=pretrained, progress=False)

        for param in model.parameters():
            param.requires_grad = not freeze_backbone
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


@register_model
class MC3_18(nn.Module):

    def __init__(self,
                 num_classes=3,
                 pretrained=False,
                 freeze_backbone=False,
                 dropout_prob=0):
        super().__init__()
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        model = mc3_18(pretrained=pretrained, progress=False)

        for param in model.parameters():
            param.requires_grad = not freeze_backbone
        model.fc = nn.Linear(512, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)
