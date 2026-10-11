"""Swin-T / CNN factory with no ImageNet download (checkpoint overwrites weights)."""
import torch.nn as nn
import torchvision.models as tvm

NUM_CLASSES = 2


def build_model_no_download(name: str) -> nn.Module:
    name = name.lower()
    if name == "swin_t":
        m = tvm.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, NUM_CLASSES)
    elif name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    elif name == "resnet50":
        m = tvm.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return m
