"""Lightning Modules for ensembles of detectors."""

from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models.detection as detection_models
import xarray as xr
import yaml
from joblib import Parallel, delayed
from lightning import LightningModule

from ethology.detectors.ensembles.fusion import weighted_boxes_fusion_in_pixels
from ethology.detectors.ensembles.utils import (
    arrays_to_ds_variables,
    pad_to_max_first_dimension,
)


class EnsembleDetector(LightningModule):
    """Ensemble of (trained) detectors for inference.

    Attributes
    ----------
    config_file: str
        Path to the YAML config file.
    """

    def __init__(self, config_file: str | Path):
        super().__init__()

        # Load config
        self.config_file = Path(config_file)
        with open(self.config_file) as f:
            self.config = yaml.safe_load(f)

        # Load list of models (nn.ModuleList)
        self.list_models = self.load_models()

    def load_models(self) -> nn.ModuleList:
        """Load models from checkpoints."""
        models_config = self.config["models"]
        model_class = getattr(detection_models, models_config["model_class"])

        list_models = []
        for checkpoint_path in models_config["checkpoints"]:
            # Get model architecture and weights
            model = model_class(**models_config["model_kwargs"])
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["state_dict"]

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
            model.load_state_dict(model_state_dict)

            # Append to list
            list_models.append(model)
        return nn.ModuleList(list_models)

    def fuse_bboxes(self, images_batch, predictions_per_model: list[dict]):
        """Fuse bboxes per sample in CPU in parallel."""
        # Fuse bboxes per sample in CPU in parallel
        # Dispatch fusion tasks to executor (non-blocking)
        # if self.config["fusion"]["method"] == "wbf"

        # n_jobs = -1 means Use ALL available CPU cores
        # n_jobs = -2 means Use ALL available CPU cores except one
        n_jobs = self.config["fusion"].get("n_jobs", -1)

        # Parallel WBF fusion
        batch_size = len(images_batch)
        results_batch = Parallel(n_jobs=n_jobs)(
            delayed(weighted_boxes_fusion_in_pixels)(
                images_batch[i].shape[-2:],  # image height and width
                [
                    preds[i]["boxes"].cpu().numpy()
                    for preds in predictions_per_model
                ],  # same image across all models
                [
                    preds[i]["scores"].cpu().numpy()
                    for preds in predictions_per_model
                ],
                [
                    preds[i]["labels"].cpu().numpy()
                    for preds in predictions_per_model
                ],
                self.config["fusion"]["iou_th_ensemble"],
                self.config["fusion"]["skip_box_th"],
            )
            for i in range(batch_size)
        )  # list [(bboxes, scores, labels) * batch_size]

        fused_boxes_batch, fused_scores_batch, fused_labels_batch = (
            zip(*results_batch, strict=True) if results_batch else ([], [], [])
        )

        return fused_boxes_batch, fused_scores_batch, fused_labels_batch

    def predict_step(self, batch, batch_idx):
        """Predict step for a single batch."""
        # ------------------------------
        # Run all models in ensemble in GPU
        # TODO: can I vectorize this?
        # https://docs.pytorch.org/tutorials/intermediate/ensembling.html
        images_batch, _annotations_batch = batch
        predictions_per_model = [
            model(images_batch) for model in self.list_models
        ]  # [num_models][batch_size]

        # ------------------------------
        # Fuse bboxes per sample in CPU in parallel
        fused_boxes_batch, fused_scores_batch, fused_labels_batch = (
            self.fuse_bboxes(images_batch, predictions_per_model)
        )

        return fused_boxes_batch, fused_scores_batch, fused_labels_batch

    @staticmethod
    def format_predictions(raw_predictions):
        """Format as ethology detections dataset."""
        # Unzip data per batch
        (
            fused_boxes_per_batch,
            fused_scores_per_batch,
            fused_labels_per_batch,
        ) = zip(*raw_predictions, strict=True)  # [n_batches][batch_size]

        # Flatten across all batches
        fused_boxes = list(chain.from_iterable(fused_boxes_per_batch))
        fused_scores = list(chain.from_iterable(fused_scores_per_batch))
        fused_labels = list(chain.from_iterable(fused_labels_per_batch))

        # Pad arrays to max n of detections per image
        fused_boxes_padded = pad_to_max_first_dimension(fused_boxes)
        fused_scores_padded = pad_to_max_first_dimension(fused_scores)
        fused_labels_padded = pad_to_max_first_dimension(fused_labels)

        # Stack into arrays
        bboxes_array = np.transpose(
            np.stack(fused_boxes_padded), (0, -1, 1)
        )  # image_id, space-4, id
        scores_array = np.stack(fused_scores_padded)
        labels_array = np.stack(fused_labels_padded)

        # ------------------------------
        # Return as ethology detections dataset
        ds_variables = arrays_to_ds_variables(
            bboxes_array, scores_array, labels_array
        )
        detections_ds = xr.Dataset(data_vars=ds_variables)

        return detections_ds
