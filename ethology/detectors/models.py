"""Lightning modules for detectors."""

from itertools import chain
from typing import Any

import numpy as np
import torch
import xarray as xr
from lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision.models import get_model
from torchvision.models.detection import faster_rcnn, fcos, retinanet

from ethology.detectors.utils import (
    _corners_to_centroid_shape,
    _pad_to_max_first_dimension,
)
from ethology.validators.detections import ValidBboxDetectionsDataset
from ethology.validators.utils import _check_output

# Registry of supported models with their constructors
MODEL_REGISTRY = {
    "fasterrcnn_resnet50_fpn_v2": (faster_rcnn.fasterrcnn_resnet50_fpn_v2),
    "fasterrcnn_mobilenet_v3_large_fpn": (
        faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn
    ),
    "fcos_resnet50_fpn": fcos.fcos_resnet50_fpn,
    "retinanet_resnet50_fpn_v2": retinanet.retinanet_resnet50_fpn_v2,
}


class SingleDetector(LightningModule):
    """LightningModule implementation of Faster R-CNN for object detection.

    Parameters
    ----------
    config : dict
        Configuration of the model.

    """

    def __init__(self, config: dict[str, Any]):
        """Initialise the Faster R-CNN model with the given configuration."""
        super().__init__()
        self.config = config
        self.model = self._configure_model()

        # save all arguments passed to __init__
        self.save_hyperparameters()

    # -------- Initialise model ----------------------------
    def _configure_model(self) -> torch.nn.Module:
        """Initialise model from ckpt if provided, else from pretrained."""
        if "checkpoint" in self.config:
            model = self._configure_model_from_checkpoint(
                self.config["checkpoint"]
            )
        else:
            model = self._configure_model_pretrained()
        return model

    def _configure_model_pretrained(self) -> torch.nn.Module:
        """Initialise Faster R-CNN model from pretrained.

        Use default weights, backbone, and box predictor. Initialise
        classification head with random weights and size to the number
        of classes.

        Keeping the classification head may not be useful if the domain
        is very different to COCO or natural images, or if looking for
        fine grained detection. They may be helpful in small datasets (< 100
        images)
        """
        # Get model name and number of classes
        # Default: fasterrcnn_resnet50_fpn_v2 and 2 classes
        model_name = self.config.get(
            "model_name", "fasterrcnn_resnet50_fpn_v2"
        )
        num_classes = self.config.get("num_classes", 2)

        # Check if model is supported
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        # Load selected model with pretreained weights in backbone and head
        # For faster-rcnn: both backbone and RPN are loaded with pretrained
        # weights
        model = MODEL_REGISTRY[model_name](weights="DEFAULT")

        # Keep as much as possible from the bbox prediction head
        if "fasterrcnn" in model_name:
            # Reinitialise box predictor for the required number of classes
            # (both cls_score and bbox_pred are reinitialised)
            # Note: in Faster R-CNN, the bbox regression is class-specific;
            # it learns a different way of refining bboxes for each class.
            # So we need to reinitialise the full box predictor.
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
                in_features,
                self.config["num_classes"],
            )
        elif "retinanet" in model_name:
            # CHECK
            # In retinanet bbox regression is class-agnostic
            in_channels = model.head.classification_head.conv[0].in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = (
                retinanet.RetinaNetClassificationHead(
                    in_channels, num_anchors, num_classes
                )
            )

        elif "fcos" in model_name:
            # CHECK
            # In fcos bbox regression is class-agnostic
            in_channels = model.head.classification_head.conv[0].in_channels
            num_anchors = model.head.classification_head.num_anchors
            model.head.classification_head = fcos.FCOSClassificationHead(
                in_channels, num_anchors, num_classes
            )

        return model

    def _configure_model_from_checkpoint(
        self, checkpoint_path: str
    ) -> torch.nn.Module:
        """Initialise Faster R-CNN model from checkpoint."""
        # Get checkpoint
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)

        # Instantiate model with ckpt weights
        model = get_model(
            self.config["model_class"],  # "fasterrcnn_resnet50_fpn_v2",
            **self.config.get("model_kwargs", {}),
        )
        model_state_dict = self._get_model_state_dict(checkpoint_dict)
        model.load_state_dict(model_state_dict, strict=True)
        return model

    @staticmethod
    def _get_model_state_dict(checkpoint: dict) -> dict:
        """Get model state dict from checkpoint dictionary."""
        # Get the state_dict key if it exists,
        # otherwise use the checkpoint itself as the state dict
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Remove "model." prefix if present (if not, it leaves
        # keys unchanged).
        # Note: PyTorch Lightning saves the model with a "model."
        # prefix in the state_dict keys if you defined self.model
        # in your LightningModule
        return {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
        }

    # ------- Inference -----------------------
    def predict_step(self, batch, batch_idx) -> Any:
        """Run inference on a batch of images."""
        images_batch, _annotations_batch = batch
        raw_prediction_dicts = self.model(images_batch)

        return raw_prediction_dicts

    @_check_output(ValidBboxDetectionsDataset)
    def run_inference(
        self,
        trainer: Trainer,
        dataloader: DataLoader,
        attrs: dict | None = None,
    ) -> xr.Dataset:
        """Run inference and return the xarray dataset.

        Convenience method that wraps trainer.predict() and
        format_predictions().
        """
        predictions = trainer.predict(self, dataloader)
        return self._format_predictions(predictions, attrs=attrs)

    # ------- Formatting -------------------
    @staticmethod
    @_check_output(ValidBboxDetectionsDataset)
    def _format_predictions(
        predictions: list[dict],
        attrs: dict | None = None,
    ) -> xr.Dataset:
        """Format predictions as an ``ethology`` detections dataset."""
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
                "category": (  # labels are renamed as "category"
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
