"""Eval-only tooth-crop transform (ImageNet normalize, 224²). No training augs."""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    pad_kwargs = dict(min_height=image_size, min_width=image_size, border_mode=0)
    try:
        pad = A.PadIfNeeded(**pad_kwargs, fill=0)
    except TypeError:
        pad = A.PadIfNeeded(**pad_kwargs, value=0)
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        pad,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
