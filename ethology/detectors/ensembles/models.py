"""Lightning Modules for ensembles of detectors."""

from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import yaml
from lightning import LightningModule
from torchvision.models import detection, get_model, list_models

from ethology.detectors.ensembles.utils import pad_to_max_first_dimension
from ethology.validators.detections import ValidBboxDetectionsDataset
from ethology.validators.utils import _check_output


class EnsembleDetector(LightningModule):
    """Ensemble of (trained) detectors for inference.

    Attributes
    ----------
    config_file: str
        Path to the YAML config file.

    """

    def __init__(self, config_file: str | Path):
        """Initialise ensemble of detectors."""
        super().__init__()

        # Load config
        self.config_file = Path(config_file)
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)

        # Run checks
        self._validate_model_class(self.config["models"]["model_class"])

        # Load list of models (nn.ModuleList)
        self.list_models = self._load_models()

    @staticmethod
    def _validate_model_class(model_class_str: str) -> None:
        """Validate that the model is part of torchvision.models.detection."""
        valid_models = set(list_models(module=detection))
        if model_class_str not in valid_models:
            valid_sorted = ", ".join(sorted(valid_models))
            raise ValueError(
                f"'{model_class_str}' is not a supported detection model. "
                f"Valid options: {valid_sorted}"
            )

    def _load_models(self) -> nn.ModuleList:
        """Load models from checkpoints."""
        # Get model config
        models_config = self.config["models"]

        # Load weights
        list_models = []
        for checkpoint_path in models_config["checkpoints"]:
            # Get checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Instantiate model with ckpt weights
            model = get_model(
                models_config["model_class"],
                **models_config.get("model_kwargs", {}),
            )
            model_state_dict = self._get_model_state_dict(checkpoint)
            model.load_state_dict(model_state_dict, strict=True)

            # Append model to list
            list_models.append(model)

        return nn.ModuleList(list_models)

    @staticmethod
    def _get_model_state_dict(checkpoint):
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
        # PyTorch Lightning saves the model with a "model."
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

    def predict_step(self, batch, batch_idx):
        """Predict step for a single batch."""
        # ------------------------------
        # Run all models in ensemble in GPU
        images_batch, _annotations_batch = batch
        raw_prediction_dicts_per_model = [
            model(images_batch) for model in self.list_models
        ]  # [num_models][batch_size]

        # Transpose to [batch_size][num_models] for easier downstream
        # processing
        raw_prediction_dicts_per_sample = [
            list(one_sample_all_models)
            for one_sample_all_models in zip(
                *raw_prediction_dicts_per_model, strict=True
            )
        ]  # [batch_size][num_models]

        return raw_prediction_dicts_per_sample

    @_check_output(ValidBboxDetectionsDataset)
    def format_predictions(self, attrs: dict | None = None) -> xr.Dataset:
        """Format as ethology detections dataset with model axis."""
        # Get results from trainer
        raw_predictions_per_model = self.trainer.predict_loop.predictions

        # Flatten batches
        raw_prediction_dicts_per_sample = list(
            chain.from_iterable(raw_predictions_per_model)
        )  # [sample][model]

        # Parse output from dicts
        output_per_sample: dict[str, list] = {
            "boxes": [],
            "scores": [],
            "labels": [],
        }
        for ky in output_per_sample:
            output_per_sample[ky] = [
                [sample[m][ky] for m in range(len(self.list_models))]
                for sample in raw_prediction_dicts_per_sample
            ]  # [sample][model]

        # Pad across models and across image_ids
        fill_value = {"boxes": np.nan, "scores": np.nan, "labels": -1}
        output_per_sample_padded: dict[str, list] = {
            ky: [] for ky in output_per_sample
        }
        for ky in output_per_sample_padded:
            output_per_sample_padded[ky] = pad_to_max_first_dimension(
                [
                    # pad across models
                    np.stack(
                        pad_to_max_first_dimension(
                            output_one_sample, fill_value[ky]
                        ),
                        axis=-1,
                    )
                    for output_one_sample in output_per_sample[ky]
                ],
                fill_value[ky],
            )

        # Stack and reorder dimensions
        bboxes_array = np.transpose(
            np.stack(output_per_sample_padded["boxes"]),
            (0, -2, 1, -1),
        )
        scores_array = np.stack(output_per_sample_padded["scores"])
        labels_array = np.stack(output_per_sample_padded["labels"])
        # arrays of shape (image_id, 4/1, n_max_detections, n_models)

        # Compute centroid and shape arrays
        centroid_array = 0.5 * (bboxes_array[:, 0:2] + bboxes_array[:, 2:4])
        shape_array = bboxes_array[:, 2:4] - bboxes_array[:, 0:2]

        # Return as ethology detections dataset
        max_n_detections = bboxes_array.shape[-2]
        n_images = bboxes_array.shape[0]

        return xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "id", "model"],
                    centroid_array,
                ),
                "shape": (["image_id", "space", "id", "model"], shape_array),
                "confidence": (["image_id", "id", "model"], scores_array),
                "label": (["image_id", "id", "model"], labels_array),
            },
            coords={
                "image_id": np.arange(n_images),
                "space": ["x", "y"],
                "id": np.arange(max_n_detections),
                "model": np.arange(len(self.list_models)),
            },
            attrs=attrs if attrs else {},
        )
