import pytest
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

from ethology.datasets.inference import (
    InferenceImageDataset,
    get_default_inference_transforms,
)


def test_get_default_inference_transforms():
    """Test default transforms produce expected output."""
    out_transforms = get_default_inference_transforms()

    # Create a dummy image and transform it
    dummy_img = Image.new(
        "RGB",
        (10, 10),
        color=(128, 128, 128),
    )
    result = out_transforms(dummy_img)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert result.min() >= 0.0 and result.max() <= 1.0  # scaled


class TestInferenceImageDataset:
    """Tests for InferenceImageDataset."""

    @pytest.fixture
    def sample_images_dir(self, tmp_path):
        """Create a temporary directory with 3 black images."""
        list_images = []
        for i in range(3):
            img = Image.new("RGB", (200, 100))  # width, height
            output_path = tmp_path / f"img_{i:02d}.png"
            img.save(output_path)
            list_images.append(output_path)
        return tmp_path, list_images

    def test_len(self, sample_images_dir):
        """Test dataset length matches number of images."""
        images_dir_path, list_images = sample_images_dir
        dataset = InferenceImageDataset(
            images_dir=images_dir_path, file_pattern="*.png"
        )
        assert len(dataset) == len(list_images)

    def test_getitem(self, sample_images_dir):
        """Test __getitem__ returns (image, empty dict)."""
        images_dir_path, _ = sample_images_dir
        dataset = InferenceImageDataset(
            images_dir=images_dir_path,
            file_pattern="*.png",
            transforms=get_default_inference_transforms(),
        )
        # Take one sample
        img, annots = dataset[0]

        # Check outputs
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 100, 200)  # C, H, W
        assert annots == {}

    def test_images_sorted(self, sample_images_dir):
        """Test images in dataset are in alphabetical order."""
        images_dir_path, list_files_images_dir = sample_images_dir
        dataset = InferenceImageDataset(
            images_dir=images_dir_path,
            file_pattern="*.png",
        )

        list_filenames = [f.name for f in dataset.image_files]
        assert list_filenames == sorted(
            [f.name for f in list_files_images_dir]
        )

    def test_file_pattern_filters(self, sample_images_dir):
        """Test that file_pattern correctly filters files."""
        images_dir_path, _ = sample_images_dir

        # Add a jpg file to the dataset directory to filter out
        img_jpg = Image.new("RGB", (200, 100))
        img_jpg.save(images_dir_path / "test.jpg")

        # Build a dataset from that directory with png filter
        dataset = InferenceImageDataset(
            images_dir=images_dir_path,
            file_pattern="*.png",
        )

        # Check there are no jpg files
        assert not all([im.suffix == ".jpg" for im in dataset.image_files])

    def test_default_transforms(self, sample_images_dir):
        """Check that default transforms are assigned if None specified."""
        # Create a minimal dataset
        images_dir_path, _ = sample_images_dir
        dataset = InferenceImageDataset(
            images_dir=images_dir_path,
            file_pattern="*.png",
        )

        # Get one sample
        img, _annot = dataset[0]

        # Check transforms are applied as expected
        assert isinstance(img, torch.Tensor)
        assert img.dtype == torch.float32
        assert img.max() <= 1.0  # scaled from [0,255] to [0,1]

        # Check type
        assert isinstance(dataset.transforms, transforms.Compose)
        assert len(dataset.transforms.transforms) == 2
