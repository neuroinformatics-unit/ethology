"""Base dataset module for loading annotated images."""

from pathlib import Path
from typing import Optional, Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class BaseImageDataset(Dataset):
    """Basic dataset for loading annotated images.
    
    Parameters
    ----------
    annotations_df : pd.DataFrame
        DataFrame containing annotations with required columns:
        'image_filename', 'image_id'
    images_dir : str | Path
        Directory containing the images
    transform : callable, optional
        Optional transform to be applied to images
    """
    
    def __init__(
        self,
        annotations_df: pd.DataFrame,
        images_dir: str | Path,
        transform: Optional[Callable] = None
    ):
        self.annotations = annotations_df
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Get unique image IDs and filenames
        self.image_data = (
            self.annotations[["image_id", "image_filename"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_data)
    
    def __getitem__(self, idx: int) -> dict:
        """Get an image and its metadata.
        
        Parameters
        ----------
        idx : int
            Index of the image to get
            
        Returns
        -------
        dict
            Dictionary containing:
                - image: PIL Image or transformed image
                - image_id: unique identifier for the image
                - filename: original image filename
        """
        # Get image info
        image_info = self.image_data.iloc[idx]
        image_id = image_info["image_id"]
        filename = image_info["image_filename"]
        
        # Load image
        image_path = self.images_dir / filename
        image = Image.open(image_path)
        
        # Apply transforms if any
        if self.transform is not None:
            image = self.transform(image)
            
        return {
            "image": image,
            "image_id": image_id,
            "filename": filename
        }
    