"""Datasets and related utilities for inference without ground-truth."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset


class InferenceImageDataset(Dataset):
    """A simple dataset for images with no ground-truth.

    It returns a dummy annotations dictionary.
    """

    def __init__(self, root_dir, file_pattern, transform=None):
        """Initialise the dataset."""
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(self.root_dir.glob(file_pattern))

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Return the image and the dummy annotations dictionary."""
        img_path = Path(self.root_dir) / self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, {}  # return a dummy annotations dict


def get_default_inference_transforms() -> transforms.Compose:
    """Return the default transforms for inference."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
