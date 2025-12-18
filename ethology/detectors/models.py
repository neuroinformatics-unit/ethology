"""Lightning modules for detectors."""

import difflib
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
    _pad_to_max_first_dimension,
    corners_to_centroid_shape,
)
from ethology.validators.detections import ValidBboxDetectionsDataset
from ethology.validators.utils import _check_output

# Registry of supported models with their constructors
MODEL_CONSTRUCTORS_REGISTRY = {
    "fasterrcnn_resnet50_fpn_v2": (faster_rcnn.fasterrcnn_resnet50_fpn_v2),
    "fasterrcnn_mobilenet_v3_large_fpn": (
        faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn
    ),
    "fcos_resnet50_fpn": fcos.fcos_resnet50_fpn,
    "retinanet_resnet50_fpn_v2": retinanet.retinanet_resnet50_fpn_v2,
}

DEFAULT_NUM_CLASSES = 91


class ObjectDetector(LightningModule):
    """LightningModule for object detection using torchvision models.

    Supports Faster R-CNN, RetinaNet, and FCOS architectures.
    This module is intended for inference only.

    Parameters
    ----------
    config : dict
        Configuration of the model, with the expected keys:
        - "model_class": str
            Name of the model to initialise. Should be one of:
            - ``fasterrcnn_resnet50_fpn_v2``,
            - ``fasterrcnn_mobilenet_v3_large_fpn``,
            - ``fcos_resnet50_fpn``,
            - ``retinanet_resnet50_fpn_v2``.
        - "model_kwargs": dict
            Keyword arguments to pass to the model constructor. See
            the `torchvision.models.detection module <https://docs.pytorch.org/vision/main/models.html#object-detection>`_
            for further details.
        - "checkpoint": str | None
            Path to the trained model checkpoint. If ``None``, the model is
            initialised from pretrained weights.

    Attributes
    ----------
    config : dict
        The configuration dictionary passed to the constructor.
    model : torch.nn.Module
        The object detector model.

    Examples
    --------
    Initialise a Faster R-CNN model from pretrained weights:

    >>> from ethology.detectors.models import ObjectDetector
    >>> config = {
    ...     "model_class": "fasterrcnn_resnet50_fpn_v2",
    ...     "model_kwargs": {
    ...         "n_classes": 2,
    ...         "weights": None,
    ...         "weights_backbone": None,
    ...     },
    ...     "checkpoint": "/path/to/checkpoint.ckpt",
    ... }
    >>> model = ObjectDetector(config)

    """

    def __init__(self, config: dict[str, Any]):
        """Initialise object detector for the given configuration."""
        super().__init__()
        self.config = self._validate_config(config)
        self.model = self._configure_model()

        # save all arguments passed to __init__ to
        # hparams attribute
        self.save_hyperparameters()

    # --------- Validate config -----------------
    @staticmethod
    def _validate_config(config):
        """Validate config dict for detector."""
        # Check config is a dictionary
        if not isinstance(config, dict):
            raise TypeError(
                "config must be a dictionary",
                f"but got {type(config).__name__}",
            )

        # model_class should always be defined
        if "model_class" not in config:
            raise ValueError("model_class must be defined in config")

        # model_class required when loading from checkpoint
        # if "checkpoint" in config and "model_class" not in config:
        #     raise ValueError(
        #         "model_class must be defined in config "
        #         "when loading from checkpoint"
        #     )

        # Check if model_class is supported if defined
        if config["model_class"] not in MODEL_CONSTRUCTORS_REGISTRY:
            model_class = config["model_class"]
            raise ValueError(
                f"Model '{model_class}' not supported. "
                f"Available: {list(MODEL_CONSTRUCTORS_REGISTRY.keys())}"
            )

        # Check model_kwargs type is dict if provided
        if "model_kwargs" in config:
            if not isinstance(config["model_kwargs"], dict):
                raise TypeError(
                    f"model_kwargs must be a dict, got "
                    f"{type(config['model_kwargs']).__name__}"
                )

            # Check for keys similar to 'num_classes' but not exact
            list_fuzzy_matches = ["num_classes"]
            for key in config["model_kwargs"]:
                if key not in list_fuzzy_matches:
                    close_matches = difflib.get_close_matches(
                        key, list_fuzzy_matches
                    )
                    if close_matches:
                        raise ValueError(
                            f"Invalid key '{key}' in model_kwargs. "
                            f"Did you mean '{close_matches[0]}'?"
                        )

        return config

    # -------- Initialise model ----------------------------
    def _configure_model(self) -> torch.nn.Module:
        """Initialise model from ckpt if provided, else from pretrained."""
        # Extract model params as attributes
        self.model_class = self.config.get("model_class")
        self.model_kwargs = self.config.get("model_kwargs", {})

        # Set num_classes to default if not set
        if "num_classes" not in self.model_kwargs:
            self.model_kwargs["num_classes"] = DEFAULT_NUM_CLASSES

        # TODO: log model params?

        # Delegate to the appropriate function
        if "checkpoint" not in self.config:
            model = self._configure_model_pretrained()
        else:
            model = self._configure_model_from_checkpoint(
                self.config["checkpoint"]
            )

        return model

    def _configure_model_pretrained(self) -> torch.nn.Module:
        """Load pretrained weights into model.

        Default weights are used when possible. If there is a shape mismatch
        in the layers, the weights are reinitialised.

        Returns
        -------
        torch.nn.Module
            The initialised object detector model.

        Notes
        -----
        Keeping the classification head may not be useful if the domain
        is very different to COCO or natural images, or if looking for
        fine grained detection. They may be helpful in small datasets
        (< 100 images)

        """
        # Load selected model with pretreained weights in backbone and head
        model = MODEL_CONSTRUCTORS_REGISTRY[self.model_class](
            weights="DEFAULT"
        )

        # Adapt model if there is a mismatch with the requested number of
        # classes
        n_classes_model = _get_n_classes_in_detector(
            model, self.model_class
        )  # shape of loaded model
        if self.model_kwargs["num_classes"] != n_classes_model:
            # Keep as much as possible from the bbox prediction head
            if "fasterrcnn" in self.model_class:
                # Reinitialise box predictor for the required number of classes
                # (both cls_score and bbox_pred are reinitialised)
                # Note: in Faster R-CNN, the bbox regression is class-specific;
                # it learns a different way of refining bboxes for each class.
                # So we need to reinitialise the full box predictor if the
                # number of classes is different from COCO2017.
                in_features = (
                    model.roi_heads.box_predictor.cls_score.in_features
                )
                model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
                    in_features,
                    self.model_kwargs["num_classes"],
                )
            elif "retinanet" in self.model_class:
                # In retinanet bbox regression is class-agnostic, so we can
                # retain it
                in_channels = model.head.classification_head.conv[0][
                    0
                ].in_channels
                num_anchors = model.head.classification_head.num_anchors
                model.head.classification_head = (
                    retinanet.RetinaNetClassificationHead(
                        in_channels,
                        num_anchors,
                        self.model_kwargs["num_classes"],
                    )
                )

            elif "fcos" in self.model_class:
                # In fcos bbox regression is class-agnostic, so we can retain
                # it
                in_channels = model.head.classification_head.conv[
                    0
                ].in_channels
                num_anchors = model.head.classification_head.num_anchors
                model.head.classification_head = fcos.FCOSClassificationHead(
                    in_channels,
                    num_anchors,
                    self.model_kwargs["num_classes"],
                )

        return model

    def _configure_model_from_checkpoint(
        self, checkpoint_path: str
    ) -> torch.nn.Module:
        """Load weights from checkpoint into model."""
        # Get checkpoint
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)

        # Instantiate model
        model = get_model(self.model_class, **self.model_kwargs)

        # Get state dict from checkpoint and load into model
        model_state_dict = self._get_model_state_dict(checkpoint_dict)
        model.load_state_dict(model_state_dict, strict=True)
        return model

    # ------ Convenience functions --------------
    @staticmethod
    def _get_model_state_dict(checkpoint: dict) -> dict:
        """Get model state dict from checkpoint dictionary.

        Returns
        -------
        dict
            The model state dictionary.

        """
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
    def predict_step(
        self, batch: tuple[torch.Tensor, dict], batch_idx: int
    ) -> dict:
        """Run inference on a batch of images.

        Parameters
        ----------
        batch : tuple[torch.Tensor, dict]
            A tuple containing the batch of images and the corresponding
            annotations.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict
            The raw predictions as a dictionary with the keys:
            - "boxes": torch.Tensor
            - "scores": torch.Tensor
            - "labels": torch.Tensor

        """
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
        """Run inference on the input dataloader.

        Convenience method that wraps ``trainer.predict()`` and
        ``_format_predictions()`` and returns the formatted predictions as an
        ``ethology`` detections dataset.

        Parameters
        ----------
        trainer : lightning.Trainer
            The trainer to use for inference.
        dataloader : torch.utils.data.DataLoader
            The dataloader to use for inference.
        attrs : dict | None
            Attributes to add to the predictions dataset.

        Returns
        -------
        xr.Dataset
            The formatted predictions as an ``ethology`` detections dataset.

        """
        predictions = trainer.predict(self, dataloader)
        return self._format_predictions(predictions, attrs=attrs)

    # ------- Formatting -------------------
    @staticmethod
    @_check_output(ValidBboxDetectionsDataset)
    def _format_predictions(
        predictions: list[list[dict]],
        attrs: dict | None = None,
    ) -> xr.Dataset:
        """Format predictions as an ``ethology`` detections dataset.

        Empty batches are ignored. Outer list is batches, inner list is images.
        """
        # Check input data
        if not isinstance(predictions, list):
            raise TypeError(
                f"predictions must be a list, got {type(predictions).__name__}"
            )
        if len(predictions) == 0:
            raise ValueError(
                "predictions list is empty. "
                "Cannot format an empty predictions list."
            )

        # Flatten output predictions
        predictions_dict_per_img = list(chain.from_iterable(predictions))

        # Check flattened data
        if len(predictions_dict_per_img) == 0:
            raise ValueError(
                "No predictions to format. "
                "predictions list contains no image data."
            )

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
        centroid_array, shape_array = corners_to_centroid_shape(
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
                "category": (  # labels are renamed as "category" array
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


def _get_n_classes_in_detector(
    model: torch.nn.Module, model_class: str
) -> int:
    """Extract the number of classes from model based on its architecture.

    Parameters
    ----------
    model : torch.nn.Module
        The object detector model.
    model_class : str
        Name of the model architecture (e.g., "fasterrcnn_resnet50_fpn_v2").

    Returns
    -------
    int
        The number of classes the model is configured to detect.

    Raises
    ------
    ValueError
        If the model architecture is not supported.

    """
    if "fasterrcnn" in model_class:
        return model.roi_heads.box_predictor.cls_score.out_features
    elif any(x in model_class for x in ["retinanet", "fcos"]):
        cls_head = model.head.classification_head
        return cls_head.cls_logits.out_channels // cls_head.num_anchors
    else:
        raise ValueError(
            f"Could not retrieve the number of classes. "
            f"Unsupported model class: {model_class}"
        )
