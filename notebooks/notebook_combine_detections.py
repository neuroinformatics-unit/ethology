# Compute ensemble of detections
# See
# - https://docs.pytorch.org/tutorials/intermediate/ensembling.html
# - https://discuss.pytorch.org/t/how-to-make-predictions-using-an-ensemble-of-models-in-parallel-on-a-single-gpu/202412/4

# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from ensemble_boxes import weighted_boxes_fusion
from torch.utils.data import random_split

from ethology.datasets.create import create_coco_dataset
from ethology.detectors.inference import _detections_per_image_id_as_ds

xr.set_options(display_expand_attrs=False)

# %matplotlib widget

# %%
# Input data


# models
list_models = [
    # "above_0th", -- skip for now because diff image_ids
    "above_1st",
    "above_5th",
    "above_10th",
    "above_25th",
    "above_50th",
]
timestamp_ref = "20250717_115247"
predictions_dir = Path(
    "/home/sminano/swc/project_ethology/remove_small_bboxes_inD_output"
)
model_to_path = {
    model_key: predictions_dir
    / f"{model_key}_detections_val_set_seed_42_{timestamp_ref}.nc"
    for model_key in list_models
}


# dataset
dataset_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full")
annotations_dir = Path("/home/sminano/swc/project_ethology/large_annotations")
full_gt_annotations_file = (
    annotations_dir / "VIA_JSON_combined_coco_gen_sorted_imageIDs.json"
)
image_width = 4096  # pixels
image_height = 2160  # pixels


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load dataset from full GT data


def split_dataset_crab_repo(dataset_coco, seed_n, config):
    """Split dataset like in crabs repo."""
    # Split data into train and test-val sets
    rng_train_split = torch.Generator().manual_seed(seed_n)
    rng_val_split = torch.Generator().manual_seed(seed_n)

    # Split train and test-val sets
    train_dataset, test_val_dataset = random_split(
        dataset_coco,
        [config["train_fraction"], 1 - config["train_fraction"]],
        generator=rng_train_split,
    )

    # Split test/val sets from the remainder
    test_dataset, val_dataset = random_split(
        test_val_dataset,
        [
            1 - config["val_over_test_fraction"],
            config["val_over_test_fraction"],
        ],
        generator=rng_val_split,
    )

    print(f"Seed: {seed_n}")
    print(f"Number of training samples: {len(train_dataset)}")  # images
    print(f"Number of validation samples: {len(val_dataset)}")  # images
    print(f"Number of test samples: {len(test_dataset)}")  # images

    return train_dataset, val_dataset, test_dataset


inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create COCO dataset
dataset_coco = create_coco_dataset(
    images_dir=Path(dataset_dir) / "frames",
    annotations_file=full_gt_annotations_file,
    composed_transform=inference_transforms,
)

# Split dataset like in crabs repo
train_dataset, val_dataset, test_dataset = split_dataset_crab_repo(
    dataset_coco,
    seed_n=42,
    config={"train_fraction": 0.8, "val_over_test_fraction": 0.5},
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read detections

ds_predictions_per_model = {
    model_key: xr.open_dataset(model_to_path[model_key])
    for model_key in list_models
}

# %%
# Get list of image_ids
list_image_ids = (
    ds_predictions_per_model[list_models[0]].coords["image_id"].values.tolist()
)

# check the same image_ids are present in all models
assert np.all(
    [
        ds_predictions_per_model[model_key].coords["image_id"].values.tolist()
        == list_image_ids
        for model_key in list_models[1:]
    ]
)

# check image_ids match val dataset
list_image_ids_val_set = [annot["image_id"] for _, annot in val_dataset]
assert np.all(set(list_image_ids) == set(list_image_ids_val_set))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute consensus detections

model_weights = [1.0 / len(list_models) for _ in list_models]
iou_thr = 0.5  # threshold for a match
skip_box_thr = 0.0001  # skip boxes with confidence below this threshold
sigma = 0.1

detections_per_image_id = {}

# for image_id in list_image_ids[:1]:
for image, annots in [val_dataset[i] for i in range(10)]:
    # Get image_id
    image_id = annots["image_id"]

    # Get predictions per modelfor this image_id
    list_bboxes, list_labels, list_scores = [], [], []
    for model_key in list_models:
        # get predictions for this model and image_id
        position = (
            ds_predictions_per_model[model_key]
            .centroids.sel(image_id=image_id)
            .T
        )
        shape = (
            ds_predictions_per_model[model_key].shape.sel(image_id=image_id).T
        )
        labels = (
            ds_predictions_per_model[model_key].label.sel(image_id=image_id).T
        )
        confidence = (
            ds_predictions_per_model[model_key]
            .confidence.sel(image_id=image_id)
            .T
        )

        # normalize coordinates to [0, 1]
        img_width_height = np.array([image_width, image_height])[None, :]
        x1_y1_norm = (position - shape / 2) / img_width_height
        x2_y2_norm = (position + shape / 2) / img_width_height
        x1_y1_x2_y2_norm = np.c_[x1_y1_norm, x2_y2_norm]

        # append to list
        list_bboxes.append(x1_y1_x2_y2_norm)
        list_labels.append(labels)
        list_scores.append(confidence)

    # compute soft nms
    # ensemble_x1_y1_x2_y2_norm, ensemble_scores, ensemble_labels = soft_nms(
    ensemble_x1_y1_x2_y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            list_bboxes,
            list_scores,
            list_labels,
            # weights=model_weights,
            iou_thr=iou_thr,
            # sigma=sigma,
            skip_box_thr=skip_box_thr,
        )
    )

    # add ensemble results to dict
    ensemble_x1_y1_x2_y2 = ensemble_x1_y1_x2_y2_norm * np.tile(
        img_width_height, (1, 2)
    )
    detections_per_image_id[image_id] = {
        "boxes": ensemble_x1_y1_x2_y2,
        "scores": ensemble_scores,
        "labels": ensemble_labels,
    }

    # ------------------------------------------------------------
    # plot
    plt.figure(figsize=(10, 10))
    plt.imshow(image.permute(1, 2, 0).numpy())

    # plot GT annotations as green boxes
    for gt_box in annots["boxes"]:
        plt.gca().add_patch(
            plt.Rectangle(
                (gt_box[0], gt_box[1]),
                gt_box[2] - gt_box[0],
                gt_box[3] - gt_box[1],
                fill=False,
                edgecolor=(0, 1, 0),
                linewidth=1,
            )
        )

    # plot ensemble detections as red boxes
    for pred_box in ensemble_x1_y1_x2_y2:
        plt.gca().add_patch(
            plt.Rectangle(
                (pred_box[0], pred_box[1]),
                pred_box[2] - pred_box[0],
                pred_box[3] - pred_box[1],
                fill=False,
                edgecolor="r",
                linewidth=1,
            )
        )
    plt.title(f"Image {image_id}")
    # plt.show()

# %%
# ensemble results as a xarray dataset

ensemble_ds = _detections_per_image_id_as_ds(detections_per_image_id)


ensemble_ds.attrs["models"] = list_models
# ensemble_ds.attrs["iou_thr"] = iou_thr
# ensemble_ds.attrs["skip_box_thr"] = skip_box_thr
# ensemble_ds.attrs["sigma"] = sigma
# ensemble_ds.attrs["model_weights"] = model_weights

# save ensemble results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ensemble_ds.to_netcdf(
    predictions_dir / f"ensemble_detections_val_set_seed_42_{timestamp}.nc"
)
# %%
