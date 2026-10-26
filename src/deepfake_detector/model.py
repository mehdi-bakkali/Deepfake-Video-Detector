"""Model construction and image preprocessing."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(pretrained: bool = True) -> nn.Module:
    """Create an EfficientNet-B0 with a single fake-vs-real logit."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def make_transform(training: bool = False) -> transforms.Compose:
    """Build ImageNet-normalized transforms for sampled video frames."""
    operations: list[object] = [transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
            ]
        )
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(operations)


def select_device() -> torch.device:
    """Choose CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
