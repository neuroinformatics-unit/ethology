# Compute ensemble of detections
# See
# - https://docs.pytorch.org/tutorials/intermediate/ensembling.html
# - https://discuss.pytorch.org/t/how-to-make-predictions-using-an-ensemble-of-models-in-parallel-on-a-single-gpu/202412/4

# %%
import json
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from ensemble_boxes import weighted_boxes_fusion
from torch.utils.data import random_split

from ethology.datasets.create import create_coco_dataset
from ethology.detectors.evaluate import evaluate_detections_hungarian
from ethology.detectors.inference import _detections_per_image_id_as_ds

xr.set_options(display_expand_attrs=False)

# %matplotlib widget

# %%
# Helper function for plotting and saving ensemble detections


def plot_and_save_ensemble_detections(
    image,
    gt_boxes_x1_y1_x2_y2,
    pred_boxes_x1_y1_x2_y2,
    pred_boxes_scores,
    image_id,
    output_dir,
    precision,
    recall,
):
    """Plot ground truth and ensemble detections on image and save as PNG."""
    # Convert tensor to numpy array and transpose from (C, H, W) to (H, W, C)
    # Convert from float [0,1] to uint8 [0,255] for OpenCV
    # Convert from RGB to BGR for OpenCV
    image_cv = image.permute(1, 2, 0).numpy()
    image_cv = (image_cv * 255).astype(np.uint8)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # plot GT annotations as green boxes
    for gt_box in gt_boxes_x1_y1_x2_y2:
        x1, y1, x2, y2 = gt_box.cpu().numpy().astype(int)
        cv2.rectangle(
            image_cv,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),  # Green color in BGR
            2,  # Line thickness
        )

    # plot ensemble detections as red boxes
    for pred_box, confidence in zip(
        pred_boxes_x1_y1_x2_y2, pred_boxes_scores, strict=True
    ):
        x1, y1, x2, y2 = pred_box.astype(int)

        cv2.rectangle(
            image_cv,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),  # Red color in BGR
            2,  # Line thickness
        )

        # add text with confidence score
        text = f"{confidence:.2f}"
        cv2.putText(
            image_cv,
            text,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font scale
            (0, 0, 255),  # Red color in BGR
            2,  # Line thickness
        )

    # add text with precision and recall to bottom right corner
    cv2.putText(
        image_cv,
        f"Precision: {precision:.2f}",
        (image_cv.shape[1] - 400, image_cv.shape[0] - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,  # Font scale
        (0, 0, 255),  # Red color in BGR
        4,  # Line thickness
    )
    cv2.putText(
        image_cv,
        f"Recall: {recall:.2f}",
        (image_cv.shape[1] - 400, image_cv.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,  # Font scale
        (0, 0, 255),  # Red color in BGR
        4,  # Line thickness
    )

    # Save the image as PNG
    output_filename = output_dir / f"val_set_{image_id:06d}.png"
    cv2.imwrite(str(output_filename), image_cv)
    print(f"Saved: {output_filename}")

    return image_cv


# %%
# Input data
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

# output directory
output_dir = Path("/home/sminano/swc/project_ethology/ensemble_detections")
output_dir.mkdir(parents=True, exist_ok=True)

flag_save_images = True

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
# Read detections

ds_predictions_per_model = {
    model_key: xr.open_dataset(model_to_path[model_key])
    for model_key in list_models
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get data from predictions

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

# get common seed
seed_n = np.unique(
    [ds.attrs["seed_n"] for ds in ds_predictions_per_model.values()]
)
assert seed_n.shape == (1,)
seed_n = seed_n.item()

# get common config
config_split = [
    json.loads(ds.attrs["config"]) for ds in ds_predictions_per_model.values()
]
config_split = [
    {
        "train_fraction": cfg["train_fraction"],
        "val_over_test_fraction": cfg["val_over_test_fraction"],
    }
    for cfg in config_split
]
assert all([cfg == config_split[0] for cfg in config_split[1:]])
config_split = config_split[0]


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create COCO dataset
print(full_gt_annotations_file)
dataset_coco = create_coco_dataset(
    images_dir=Path(dataset_dir) / "frames",
    annotations_file=full_gt_annotations_file,
    composed_transform=inference_transforms,
)

# Split dataset like in crabs repo
train_dataset, val_dataset, test_dataset = split_dataset_crab_repo(
    dataset_coco,
    seed_n=seed_n,
    config=config_split,
)


# check image_ids match image_ids in val dataset
list_image_ids_val_set = [annot["image_id"] for _, annot in val_dataset]
assert np.all(set(list_image_ids) == set(list_image_ids_val_set))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute consensus detections

# model_weights = [1.0 / len(list_models) for _ in list_models]
iou_thr_ensemble = 0.5  # threshold for a match
skip_box_thr = 0.0001  # skip boxes with confidence below this threshold
sigma = 0.1

detections_per_image_id = {}
precision_recall_per_sample = {}

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_output_dir = output_dir / timestamp
timestamped_output_dir.mkdir(parents=True, exist_ok=True)

if flag_save_images:
    (timestamped_output_dir / "frames").mkdir(parents=True, exist_ok=True)

iou_threshold_precision = 0.1  # threshold for a TP


# Loop thru samples in val set
for k, (image, annots) in enumerate(val_dataset):

    # Get predictions per model for this image_id
    image_id = annots["image_id"]
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

    # compute weighted boxes fusion
    # sometimes returns nan?
    ensemble_x1_y1_x2_y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            list_bboxes,
            list_scores,
            list_labels,
            # weights=model_weights,
            iou_thr=iou_thr_ensemble,
            # sigma=sigma,
            skip_box_thr=skip_box_thr,
        )
    )

    # remove rows with nan coordinates
    slc_nan_rows = np.any(np.isnan(ensemble_x1_y1_x2_y2_norm), axis=1)
    ensemble_x1_y1_x2_y2_norm = ensemble_x1_y1_x2_y2_norm[~slc_nan_rows]
    ensemble_scores = ensemble_scores[~slc_nan_rows]
    ensemble_labels = ensemble_labels[~slc_nan_rows]

    # add ensemble results to dict
    ensemble_x1_y1_x2_y2 = ensemble_x1_y1_x2_y2_norm * np.tile(
        img_width_height, (1, 2)
    )
    detections_per_image_id[image_id] = {
        "boxes": ensemble_x1_y1_x2_y2,
        "scores": ensemble_scores,
        "labels": ensemble_labels,
    }

    # compute precision per frame
    tp, fp, md = evaluate_detections_hungarian(
        ensemble_x1_y1_x2_y2, annots["boxes"], iou_threshold_precision
    )

    precision_recall_per_sample[k] = {
        "image_id": image_id,
        "precision": sum(tp) / (sum(tp) + sum(fp)),
        "recall": sum(tp) / (sum(tp) + sum(md)),
    }

    # ------------------------------------------------------------
    # plot and save ensemble detections
    if flag_save_images:
        plot_and_save_ensemble_detections(
            image=image,
            gt_boxes_x1_y1_x2_y2=annots["boxes"],
            pred_boxes_x1_y1_x2_y2=ensemble_x1_y1_x2_y2,
            pred_boxes_scores=ensemble_scores,
            image_id=image_id,
            output_dir=timestamped_output_dir / "frames",
            precision=precision_recall_per_sample[k]["precision"],
            recall=precision_recall_per_sample[k]["recall"],
        )

# %%
# average precision and recall
df_precision_recall = pd.DataFrame.from_dict(
    precision_recall_per_sample, orient="index"
)

# cast image_id to int
df_precision_recall["image_id"] = df_precision_recall["image_id"].astype(int)

print(df_precision_recall)
print(df_precision_recall.shape)
print(df_precision_recall.loc[:, ["precision", "recall"]].mean())

# add mean to df
df_precision_recall.loc["mean"] = df_precision_recall.mean()

# save as csv
df_precision_recall.to_csv(
    timestamped_output_dir / "precision_recall.csv", index=False
)


# %%
# ensemble results as a xarray dataset

ensemble_ds = _detections_per_image_id_as_ds(detections_per_image_id)
ensemble_ds.attrs["models"] = list_models

# save ensemble results
ensemble_ds.to_netcdf(
    timestamped_output_dir
    / f"ensemble_detections_val_set_seed_42_{timestamp}.nc"
)
# %%
