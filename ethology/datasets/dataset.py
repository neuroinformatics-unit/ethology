from pathlib import Path
from typing import Optional, Callable
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision

class BaseImageDataset(Dataset):
    """Base dataset module for loading annotated images for COCO-style object detection.
    
    Parameters
    ----------
    annotations_df : pd.DataFrame
        DataFrame with columns: 'image_id', 'image_filename', 'bbox', 'category_id'.
        This DataFrame should contain the COCO-style annotations.
    images_dir : str | Path
        Directory containing the images.
    transform : callable, optional
        Transform to be applied to images (e.g., torchvision transforms like Resize, ToTensor).
    """
    
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        images_dir: str | Path,
        transform: Optional[Callable] = None
    ):
        self.annotations_df = annotations_df
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Group annotations by image_id for efficient lookup
        self.grouped_annotations = annotations_df.groupby("image_id")
        
        # Get unique image metadata (image_id and image_filename)
        self.image_data = (
            annotations_df[["image_id", "image_filename"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_data)

    def __getitem__(self, idx: int) -> tuple:
        """Retrieve an image tensor and its corresponding target dictionary for detection tasks.
        
        Parameters
        ----------
        idx : int
            Index of the image to retrieve.
            
        Returns
        -------
        tuple
            A tuple containing:
                - image: Tensor representing the image.
                - target: Dictionary with the following keys:
                    'boxes' : Tensor of shape [N, 4] with bounding box coordinates.
                    'labels': Tensor of shape [N] with category labels.
                    'image_id': Tensor containing the image identifier.
        """
        # Get image metadata
        image_info = self.image_data.iloc[idx]
        image_id = image_info["image_id"]
        filename = image_info["image_filename"]
        image_path = self.images_dir / filename

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            # Default: Convert PIL image to tensor if no transform is provided
            image = torchvision.transforms.functional.to_tensor(image)

        # Get annotations for this image
        annotations = self.grouped_annotations.get_group(image_id)
        boxes = torch.tensor(annotations["bbox"].tolist(), dtype=torch.float32)
        labels = torch.tensor(annotations["category_id"].tolist(), dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id], dtype=torch.int64)
        }

        return image, target
