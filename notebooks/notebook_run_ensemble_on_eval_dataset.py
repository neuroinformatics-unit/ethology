# %%
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ethology.annotations.io import load_bboxes
from ethology.datasets.convert import torch_dataset_to_xr_dataset
from ethology.datasets.create import create_coco_dataset
from ethology.detectors.ensembles import combine_detections_across_models_wbf
from ethology.detectors.evaluate import compute_precision_recall_ds
from ethology.detectors.inference import (
    collate_fn_varying_n_bboxes,
    concat_detections_ds,
    run_detector_on_dataloader,
    # run_detector_on_dataset,
)
from ethology.detectors.load import load_fasterrcnn_resnet50_fpn_v2
from ethology.detectors.utils import (
    add_bboxes_min_max_corners,
)
from ethology.mlflow import (
    read_cli_args_from_mlflow_params,
    read_config_from_mlflow_params,
    read_mlflow_params,
)

# Set xarray options
xr.set_options(display_expand_attrs=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

dataset_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full")
annotations_dir = Path("/home/sminano/swc/project_ethology/large_annotations")
annotations_file_path = (
    annotations_dir / "VIA_JSON_combined_coco_gen_sorted_imageIDs.json"
)

experiment_ID = "617393114420881798"
ml_runs_experiment_dir = (
    Path("/home/sminano/swc/project_crabs/ml-runs") / experiment_ID
)

# I pick seed 42 for each set of models
models_dict = {
    # "above_0th": ml_runs_experiment_dir / "f348d9d196934073bece1b877cbc4d38",
    "above_1st": ml_runs_experiment_dir / "879d2f77e2b24adcb06b87d2fede6a04",
    "above_5th": ml_runs_experiment_dir / "75583ec227e3444ab692b99c64795325",
    "above_10th": ml_runs_experiment_dir / "4acc37206b1e4f679d535c837bee2c2f",
    "above_25th": ml_runs_experiment_dir / "fdcf88fcbcc84fbeb94b45ca6b6f8914",
    "above_50th": ml_runs_experiment_dir / "daa05ded0ea047388c9134bf044061c5",
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set default device: CUDA if available, otherwise mps, otherwise CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# %%
# Helper functions
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

    # remove nan predictions
    pred_boxes_x1_y1_x2_y2 = pred_boxes_x1_y1_x2_y2[
        ~np.any(np.isnan(pred_boxes_x1_y1_x2_y2), axis=1)
    ]
    pred_boxes_scores = pred_boxes_scores[~np.isnan(pred_boxes_scores)]

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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define list of models in ensemble

list_models = []
list_config = []
list_cli_args = []
for model_key in models_dict:
    # Retrieve model config and CLI args from mlflow
    trained_model_path = str(
        models_dict[model_key] / "checkpoints" / "last.ckpt"
    )

    mlflow_params = read_mlflow_params(trained_model_path)
    config = read_config_from_mlflow_params(mlflow_params)
    cli_args = read_cli_args_from_mlflow_params(mlflow_params)

    # ------------------------------------
    # Load model
    model = load_fasterrcnn_resnet50_fpn_v2(
        trained_model_path,
        num_classes=config["num_classes"],
        device=None,  # device
    )
    model.eval()
    list_models.append(model)
    list_config.append(config)
    list_cli_args.append(cli_args)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check that all models have the same dataset config
ref_config = list_config[0]
for key in ["train_fraction", "val_over_test_fraction"]:
    assert all(config[key] == ref_config[key] for config in list_config)

ref_cli_args = list_cli_args[0]
assert all(
    cli_args["seed_n"] == ref_cli_args["seed_n"] for cli_args in list_cli_args
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define common dataset for ensemble

# Define transforms for inference
inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# Create COCO dataset
dataset_coco = create_coco_dataset(
    images_dir=Path(dataset_dir) / "frames",
    annotations_file=annotations_file_path,
    composed_transform=inference_transforms,
)

# Split dataset like in crabs repo
train_dataset, val_dataset, test_dataset = split_dataset_crab_repo(
    dataset_coco,
    seed_n=ref_cli_args["seed_n"],
    config=ref_config,  # only uses train_fraction and val_over_test_fraction
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define val dataloader
# shuffle=False so that we dont shuffle the data
# after one pass over all batches
val_dataloader = DataLoader(
    val_dataset,
    batch_size=ref_config["batch_size_val"],
    shuffle=False,
    num_workers=ref_config["num_workers"],
    collate_fn=collate_fn_varying_n_bboxes,
    persistent_workers=bool(ref_config["num_workers"] > 0),
    # multiprocessing_context="fork"
    # if ref_config["num_workers"] > 0 and torch.backends.mps.is_available()
    # else None,  # see https://github.com/pytorch/pytorch/issues/87688
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute detections per model -- can I make it faster?
# can I vectorize this? (pytorch forum question)
list_detections_ds = []
for model in tqdm(list_models):
    model.to(device)

    detections_ds = run_detector_on_dataloader(
        model=model,
        dataloader=val_dataloader,
        device=device,
    )
    detections_ds = add_bboxes_min_max_corners(detections_ds)
    list_detections_ds.append(detections_ds)


# Concatenate detections across models
all_models_detections_ds = concat_detections_ds(
    list_detections_ds,
    pd.Index(range(len(list_detections_ds)), name="model"),
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fuse detections across models
fused_detections_ds = combine_detections_across_models_wbf(
    all_models_detections_ds,
    kwargs_wbf={
        "iou_thr_ensemble": 0.5,
        "skip_box_thr": 0.0001,
        "max_n_detections": 300,  # set default?
    },
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define ground truth dataset

# read annotations as a dataset
print(annotations_file_path.name)
gt_bboxes_ds = load_bboxes.from_files(annotations_file_path, format="COCO")

# fix category ID
gt_bboxes_ds["category"] = gt_bboxes_ds["category"].where(
    gt_bboxes_ds["category"] != 0, 1
)

# select only image_id in val_dataset
# Note that the max number of annotations per image in the val_dataset
# will stay as in the original dataset (also, category = -1 is not considered
# an empty value for xarrays .dropna())
list_image_ids_val = [annot["image_id"] for img, annot in val_dataset] 
gt_bboxes_val_ds = gt_bboxes_ds.sel(image_id=list_image_ids_val)


# # %%
# list_image_ids_test = [annot["image_id"] for img, annot in test_dataset]
# list_image_ids_train = [annot["image_id"] for img, annot in train_dataset]

# gt_bboxes_test_ds = gt_bboxes_ds.sel(image_id=list_image_ids_test)
# gt_bboxes_train_ds = gt_bboxes_ds.sel(image_id=list_image_ids_train)

# %%
# Alternatively: convert torch dataset into xarray detections dataset
# .....

val_ds = torch_dataset_to_xr_dataset(val_dataset)  # max -- annotations per image
test_ds = torch_dataset_to_xr_dataset(test_dataset)  # max 129 annotations per image
train_ds = torch_dataset_to_xr_dataset(train_dataset)  # max 136 annotations per image


# %%
# check data arrays are the same but with annotations in different order
# There is no guarantee that annotation with id=15 is the same in the
# xr dataset computed from the annotations file and the one computed from
# the torch dataset.
for idx in range(len(val_ds.image_id.values)):
    idcs_sorted_x1 = np.lexsort(
        (
            gt_bboxes_val_ds.position.values[idx, 1, :],
            gt_bboxes_val_ds.position.values[idx, 0, :],
        )
    )  # sort by x, then y
    idcs_sorted_x2 = np.lexsort(
        (
            val_ds.position.values[idx, 1, :],
            val_ds.position.values[idx, 0, :],
        )
    )  # sort by x, then y
    assert np.allclose(
        gt_bboxes_val_ds.position.values[idx, :, idcs_sorted_x1],
        val_ds.position.values[idx, :, idcs_sorted_x2],
        equal_nan=True,
        rtol=1e-5,
        atol=1e-8,
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate

fused_detections_ds, gt_bboxes_val_ds = compute_precision_recall_ds(
    pred_bboxes_ds=fused_detections_ds,
    gt_bboxes_ds=gt_bboxes_val_ds,
    iou_threshold=0.1,  # change to 0.5?
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot ensemble detections on a selected image

# Get first image
image_index = 0
image = val_dataset[image_index][0]
gt_annotations = val_dataset[image_index][1]

# fused_detections_ds_plot = add_bboxes_min_max_corners(fused_detections_ds)

plot_and_save_ensemble_detections(
    image=image,
    gt_boxes_x1_y1_x2_y2=gt_annotations["boxes"],
    pred_boxes_x1_y1_x2_y2=np.hstack(
        [
            fused_detections_ds[xy_corner_str]
            .isel(image_id=image_index)
            .values.T
            for xy_corner_str in ["xy_min", "xy_max"]
        ]
    ),
    pred_boxes_scores=fused_detections_ds.isel(
        image_id=image_index
    ).confidence.values,
    image_id=gt_annotations["image_id"],
    output_dir=Path.cwd(),
    precision=fused_detections_ds.isel(image_id=image_index).precision.values,
    recall=fused_detections_ds.isel(image_id=image_index).recall.values,
)
