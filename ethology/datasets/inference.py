"""Datasets and related utilities for inference without ground-truth."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset


class InferenceImageDataset(Dataset):
    """A simple dataset for images with no ground-truth annotations.

    The image files are sorted alphabetically. The annotations dictionary
    returned by ``__getitem__`` is always empty to maintain a consistent
    interface with training datasets.

    Parameters
    ----------
    images_dir
        Path to the root directory containing the images.
    file_pattern
        Pattern to match the image filenames.
    transforms
        Transforms to apply to the images. If None (default), the
        transforms from :func:`get_default_inference_transforms` are used.

    Attributes
    ----------
    images_dir : pathlib.Path
        Path to the root directory containing the images.
    transforms : torchvision.transforms.v2.Compose
        Transforms to apply to the images.
    image_files : list[pathlib.Path]
        List of paths to each of the image files, sorted
        alphabetically.

    See Also
    --------
    get_default_inference_transforms : Returns default transforms for
        inference.

    Examples
    --------
    Create a dataset from 100 ``.png`` files in the ``/path/to/images``
    directory:

    >>> from ethology.datasets.inference import InferenceImageDataset
    >>> dataset = InferenceImageDataset(
    ...     images_dir="/path/to/images",
    ...     file_pattern="*.png",
    ... )
    >>> len(dataset)
    100

    """

    def __init__(
        self,
        images_dir: Path | str,
        file_pattern: str,
        transforms: transforms.Compose | None = None,
    ):
        """Initialise dataset."""
        self.images_dir = Path(images_dir)
        self.transforms = (
            transforms
            if transforms is not None
            else get_default_inference_transforms()
        )
        self.image_files = sorted(self.images_dir.glob(file_pattern))

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Return the image and an empty annotations dictionary.

        Parameters
        ----------
        idx : int
            Index of the image to retrieve.

        Returns
        -------
        tuple[torch.Tensor, dict]
            A tuple containing the image as a tensor and an empty
            annotations dictionary.

        """
        # Open requested image
        img_path = Path(self.images_dir) / self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # If transforms are specified, apply to the image
        if self.transforms:
            image = self.transforms(image)
        return image, {}


def get_default_inference_transforms() -> transforms.Compose:
    """Return the default transforms for inference.

    Transforms the input image to a tensor and scales the pixel
    values from ``[0, 255]`` (uint8) to ``[0, 1]`` (float32).

    Returns
    -------
    torchvision.transforms.v2.Compose
        The default transforms for inference.

    """
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )
