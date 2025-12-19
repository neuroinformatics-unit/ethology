"""Datasets and related utilities for inference without ground-truth."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.utils.data import Dataset


class InferenceImageDataset(Dataset):
    """A simple dataset for images with no ground-truth annotations.

    Parameters
    ----------
    root_dir : pathlib.Path | str
        Path to the root directory containing the images.
    file_pattern : str
        Pattern to match the image filenames.
    transforms : torchvision.transforms.v2.Compose | None, optional
        Transforms to apply to the images. Default is None (i.e.,
        no transform is applied to the image).

    Attributes
    ----------
    root_dir : pathlib.Path
        Path to the root directory containing the images.
    transforms : torchvision.transforms.v2.Compose | None
        Transforms to apply to the images.
    image_files : list[pathlib.Path]
        List of paths to each of the image files, sorted
        alphabetically.

    See Also
    --------
    get_default_inference_transforms : Returns default transforms for
        inference.

    Notes
    -----
    This dataset is used for running inference on a dataset of images
    without ground-truth annotations. The image files are sorted
    alphabetically. The returned annotations dictionary is empty.

    Examples
    --------
    Create a dataset from 100 ``.png`` files in the ``/path/to/images``
    directory:

    >>> from ethology.datasets.inference import InferenceImageDataset
    >>> dataset = InferenceImageDataset(
    ...     root_dir="/path/to/images",
    ...     file_pattern="*.png",
    ... )
    >>> len(dataset)
    100

    """

    def __init__(
        self,
        root_dir: Path | str,
        file_pattern: str,
        transforms: transforms.Compose | None = None,
    ):
        """Initialise dataset."""
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.image_files = sorted(self.root_dir.glob(file_pattern))

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
        img_path = Path(self.root_dir) / self.image_files[idx]
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
