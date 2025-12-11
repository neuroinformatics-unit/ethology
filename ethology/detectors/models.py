"""Lightning modules for detectors."""

from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from lightning import LightningModule
from torchvision.models import get_model
from torchvision.models.detection import (
    faster_rcnn,
    fasterrcnn_resnet50_fpn_v2,
)

from ethology.detectors.utils import (
    _corners_to_centroid_shape,
    _pad_to_max_first_dimension,
)

# from ethology.detectors.evaluate import compute_precision_recall_in_ds
from ethology.validators.detections import ValidBboxDetectionsDataset
from ethology.validators.utils import _check_output


class SingleDetector(LightningModule):
    """LightningModule implementation of Faster R-CNN for object detection.

    Parameters
    ----------
    config : dict
        Configuration settings for the model.

    """

    def __init__(self, config: dict[str, Any]):
        """Initialise the Faster R-CNN model with the given configuration."""
        super().__init__()
        self.config = config
        self.model = self.configure_model()

        # save all arguments passed to __init__
        self.save_hyperparameters()

        # initialise metrics to log during training/val/test loop
        self.training_step_outputs = {
            "training_loss_epoch": 0.0,
            "num_batches": 0,
        }
        self.validation_step_outputs = {
            "accum_precision_epoch": 0.0,
            "accum_recall_epoch": 0.0,
            "num_batches": 0,
        }
        self.test_step_outputs = {
            "accum_precision_epoch": 0.0,
            "accum_recall_epoch": 0.0,
            "num_batches": 0,
        }

    def configure_model(self) -> torch.nn.Module:
        """Initialise model from checkpoint if provided, else from pretrained."""
        if "checkpoint" in self.config:
            model = self.configure_model_from_checkpoint(
                self.config["checkpoint"]
            )
        else:
            model = self.configure_model_pretrained()
        return model

    def configure_model_pretrained(self) -> torch.nn.Module:
        """Configure Faster R-CNN model.

        Use default weights, backbone, and box predictor. Initiliase
        classification head with random weights and size to the number
        of classes.
        """
        # TODO: expand to other torchvision detectors?
        # Load Faster-RNN with pretreained weights in backbone and head
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

        # Replace final classification layer only
        # (while keeping bbox regression weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
            in_features,
            self.config["num_classes"],
        )
        return model

    def configure_model_from_checkpoint(
        self, checkpoint_path: str
    ) -> torch.nn.Module:
        """Configure Faster R-CNN model from checkpoint."""
        # Get checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Instantiate model with ckpt weights
        model = get_model(
            "fasterrcnn_resnet50_fpn_v2",
            **self.config.get("model_kwargs", {}),
        )
        model_state_dict = self._get_model_state_dict(checkpoint)
        model.load_state_dict(model_state_dict, strict=True)
        return model

    @staticmethod
    def _get_model_state_dict(checkpoint: Path) -> dict:
        """Get model state dict from checkpoint path."""
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            # Checkpoint might be the state dict itself
            state_dict = checkpoint
        else:
            raise ValueError(
                "Checkpoint format not recognized. "
                "Expected 'state_dict' key or dict of tensors."
            )

        # Load state dict into model
        # Note: PyTorch Lightning saves the model with a "model."
        # prefix in the state_dict keys if you defined self.model
        # in your LightningModule - we remove the prefix here.
        if any(key.startswith("model.") for key in state_dict):
            model_state_dict = {
                key.replace("model.", "", 1): value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
        else:
            model_state_dict = state_dict
        return model_state_dict

    # # --------- Forward pass ---------------------
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Forward pass of the model."""
    #     return self.model(x)

    # ------- Inference -----------------------
    def predict_step(self, batch, batch_idx) -> Any:
        """Run inference on a batch of images."""
        images_batch, _annotations_batch = batch
        raw_prediction_dicts = self.model(images_batch)

        return raw_prediction_dicts

    @_check_output(ValidBboxDetectionsDataset)
    def run_inference(
        self,
        trainer,
        dataloader,
        attrs: dict | None = None,
    ) -> xr.Dataset:
        """Run inference and return the xarray dataset.

        Convenience method that combines trainer.predict() and format_predictions().
        """
        predictions = trainer.predict(self, dataloader)
        return self.format_predictions(predictions, attrs=attrs)

    # ------- Formatting -------------------
    @staticmethod
    @_check_output(ValidBboxDetectionsDataset)
    def format_predictions(
        predictions: list[dict],
        attrs: dict | None = None,
    ) -> xr.Dataset:
        """Format predictions as an ethology detections dataset."""
        # Flatten output predictions
        predictions_dict_per_img = list(chain.from_iterable(predictions))

        # Parse output from dicts
        output_per_sample = {
            key: [sample[key] for sample in predictions_dict_per_img]
            for key in ["boxes", "scores", "labels"]
        }

        # Pad across image_ids
        fill_value = {"boxes": np.nan, "scores": np.nan, "labels": -1}
        output_per_sample_padded = {
            key: np.stack(
                _pad_to_max_first_dimension(output_per_sample[key], val),
                axis=0,
            )
            for key, val in fill_value.items()
        }

        # Compute centroid and shape arrays
        bboxes_array = np.transpose(
            output_per_sample_padded["boxes"], (0, -1, 1)
        )
        centroid_array, shape_array = _corners_to_centroid_shape(
            bboxes_array[:, 0:2], bboxes_array[:, 2:4]
        )

        # Return as ethology detections dataset
        max_n_detections = bboxes_array.shape[-1]
        n_images = bboxes_array.shape[0]
        return xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "id"],
                    centroid_array,
                ),
                "shape": (["image_id", "space", "id"], shape_array),
                "confidence": (
                    ["image_id", "id"],
                    output_per_sample_padded["scores"],
                ),
                "category": (  # ------> labels are renamed as "category"!!
                    ["image_id", "id"],
                    output_per_sample_padded["labels"],
                ),
            },
            coords={
                "image_id": np.arange(n_images),
                "space": ["x", "y"],
                "id": np.arange(max_n_detections),
            },
            attrs=attrs if attrs else {},
        )
