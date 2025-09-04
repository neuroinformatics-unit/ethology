"""Utilities for loading object detectors."""

import torch
import torchvision


def load_fasterrcnn_resnet50_fpn_v2(
    trained_model_path: str,
    num_classes: int,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a Faster R-CNN ResNet50 FPN v2 detector."""
    # initialize model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )

    # load state dict
    # When you call torch.load() on a file which contains GPU tensors,
    # those tensors will be loaded to GPU by default.
    checkpoint = torch.load(trained_model_path, map_location="cpu")

    # Load weights into model
    # if model is saved with model. prefix, remove it
    if any([ky.startswith("model.") for ky in checkpoint["state_dict"]]):
        model_weights = {
            k.lstrip("model."): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        model_weights = checkpoint["state_dict"]  # ok?
    model.load_state_dict(model_weights)

    # Put model on device if provided
    if device:
        model.to(device)

    return model
