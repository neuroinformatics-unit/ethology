# %%
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from ensemble_boxes import weighted_boxes_fusion
from torch.utils.data import random_split
from tqdm import tqdm

from ethology.datasets.create import create_coco_dataset
from ethology.detectors.inference import (
    concat_detections_ds,
    run_detector_on_dataset,
)
from ethology.detectors.load import load_fasterrcnn_resnet50_fpn_v2
from ethology.detectors.utils import add_bboxes_min_max_corners
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute detections per model -- make it faster
# can I vectorize this?
# use dataloader instead?
list_detections_ds = []
for model in tqdm(list_models):
    model.to(device)
    detections_ds = run_detector_on_dataset(
        model=model,
        dataset=val_dataset,
        device=device,
    )
    detections_ds = add_bboxes_min_max_corners(detections_ds)
    list_detections_ds.append(detections_ds)


# Concatenate detections across models
all_models_detections_ds = concat_detections_ds(
    list_detections_ds,
    pd.Index(range(len(list_detections_ds)), name="model"),
)


# %%


def wbf_arrays_one_img(
    x1y1_x2y2_norm_one_img: xr.DataArray,  # ("model", "id", "space")
    scores_one_img: xr.DataArray,
    labels_one_img: xr.DataArray,
    iou_thr_ensemble,
    skip_box_thr,
):
    print(x1y1_x2y2_norm_one_img.data.shape)  # n_models, annot_id, space
    print(scores_one_img.data.shape)  # n_models, annot_id
    print(labels_one_img.data.shape)  # n_models, annot_id

    # Run WBF
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            x1y1_x2y2_norm_one_img,
            scores_one_img,
            labels_one_img,
            iou_thr=iou_thr_ensemble,
            skip_box_thr=skip_box_thr,
        )
    )

    #

    return xr.DataArray(
        data=(ensemble_x1y1_x2y2_norm * np.tile(image_width_height, (1, 2))).T,
        dims=["space", "id"],
        coords={
            "space": ["x", "y", "x", "y"],
            "id": list(range(ensemble_x1y1_x2y2_norm.shape[0])),
        },
    )

    # # Format as xarray dataset
    # # Undo x1y1 x2y2 normalization!
    # ensemble_detections_ds = detections_x1y1_x2y2_as_ds(
    #     ensemble_x1y1_x2y2_norm * np.tile(image_width_height, (1, 2)),
    #     ensemble_scores,
    #     ensemble_labels,
    # )

    # return ensemble_detections_ds


# %%
iou_thr_ensemble = 0.5
skip_box_thr = 0.0001  # skip boxes with confidence below this threshold

sel_id = 193

# compute x1y1_x2y2_norm
image_width_height = np.array(
    [
        all_models_detections_ds.attrs["image_width"],
        all_models_detections_ds.attrs["image_height"],
    ]
)
x1y1x2y2_norm = (
    xr.concat(
        [
            all_models_detections_ds["xy_min"],
            all_models_detections_ds["xy_max"],
        ],
        dim="space",
    )
    / np.tile(image_width_height, (1, 2))[None, :, :, None]
)  # model, image_id, 4, (annot) ID

x1y1x2y2_ensemble = wbf_arrays_one_img(
    x1y1x2y2_norm.sel(image_id=sel_id).transpose("model", "id", "space").data,
    all_models_detections_ds.confidence.sel(image_id=sel_id).data,
    all_models_detections_ds.label.sel(image_id=sel_id).data,
    iou_thr_ensemble=iou_thr_ensemble,
    skip_box_thr=skip_box_thr,
)

print(x1y1x2y2_ensemble.shape)


# %%
def test(
    image_id,
    x1y1x2y2_normalised,
    confidence,
    label,
):
    print(image_id)
    print(image_id.shape)
    print("---")
    iou_thr_ensemble = 0.5
    skip_box_thr = 0.0001  # skip boxes with confidence below this threshold

    x1y1x2y2_ensemble = wbf_arrays_one_img(
        x1y1x2y2_normalised,  # .sel(image_id=image_id).transpose("model", "id", "space"),
        confidence,
        label,
        iou_thr_ensemble=iou_thr_ensemble,
        skip_box_thr=skip_box_thr,
    )

    print(x1y1x2y2_ensemble.shape)
    # n_detections_in = x1y1x2y2_normalised.shape[1]
    # print(n_detections_in)
    print("<----->")

    # return x1y1x2y2_ensemble 
    # ---> could not broadcast input array from shape (4,115) into shape (4,112)


    # To have constant output dimension
    return np.pad(
        x1y1x2y2_ensemble,
        ((0, 0), (0, 300 - x1y1x2y2_ensemble.shape[1])),
        "constant",
        constant_values=np.nan,
    )

    # To have same dimensions as the input
    # n_detections_in = x1y1x2y2_normalised.shape[1]
    # n_detections_diff = (
    #     x1y1x2y2_normalised.shape[1] - x1y1x2y2_ensemble.shape[1]
    # )
    # if n_detections_diff > 0:
    #     return np.pad(
    #         x1y1x2y2_ensemble,
    #         ((0, 0), (0, n_detections_diff)),
    #         "constant",
    #         constant_values=np.nan,
    #     )
    # else:   
    #     return x1y1x2y2_ensemble[:, :n_detections_in]
    

    
# %%

x1y1_x2y2_fused = xr.apply_ufunc(
    test,
    all_models_detections_ds.image_id,
    x1y1x2y2_norm.transpose("model", "id", "space", "image_id"),
    all_models_detections_ds.confidence.transpose("model", "id", "image_id"),
    all_models_detections_ds.label.transpose("model", "id", "image_id"),
    input_core_dims=[
        [],
        ["model", "id", "space"],
        ["model", "id"],
        ["model", "id"],
    ],
    output_core_dims=[["space", "id_out"]],
    vectorize=True,
    exclude_dims={"id"},  # to allow dimensions that change size btw input and output
)


print(x1y1_x2y2_fused.shape)  # image_id, 4, padded_id








# # %%
# ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
#     weighted_boxes_fusion(
#         x1y1x2y2_norm.isel(image_id=0).transpose("model", "id", "space"),
#         all_models_detections_ds.confidence.isel(image_id=0),  # "model", "id"
#         all_models_detections_ds.label.isel(image_id=0),  # "model", "id"
#         iou_thr=iou_thr_ensemble,
#         skip_box_thr=skip_box_thr,
#     )
# )

# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Run WBF for all image IDs
# for image_id in all_models_detections_ds.image_id:
#     ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
#         weighted_boxes_fusion(
#             x1y1x2y2_norm.isel(image_id=image_id).transpose(
#                 "model", "id", "space"
#             ),
#         )
#     )


# %%
# Combine detections
# can I avoid double loop?
# should i use dataloader here too?

# Define parameters for WBF
iou_thr_ensemble = 0.5
skip_box_thr = 0.0001  # skip boxes with confidence below this threshold

(image_height, image_width) = val_dataset[0][0].shape[-2:]
image_height_width = np.array([image_width, image_height])

list_image_ids = [annot["image_id"] for img, annot in val_dataset]

detections_per_image_id = {}
for image_id in list_image_ids:
    # Get detections for current image across all models
    detections_ds_per_model = [
        ds.sel(image_id=image_id) for ds in list_detections_ds
    ]

    list_bboxes_x1y1_x2y2_norm = [
        xr.concat(
            [ds["xy_min"].T, ds["xy_max"].T],
            dim="space",
        )
        / np.tile(image_height_width, (1, 2))
        for ds in detections_ds_per_model
    ]
    list_scores = [ds.confidence.T for ds in detections_ds_per_model]
    list_labels = [ds.label.T for ds in detections_ds_per_model]

    # Run WBF
    # can I vectorize this across image_id?
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            list_bboxes_x1y1_x2y2_norm,  # n_models, n_predictions, 4
            list_scores,  # n_models, n_predictions
            list_labels,
            iou_thr=iou_thr_ensemble,
            skip_box_thr=skip_box_thr,
        )
    )

    # Remove rows with nan coordinates
    slc_nan_rows = np.any(np.isnan(ensemble_x1y1_x2y2_norm), axis=1)
    ensemble_x1y1_x2y2_norm = ensemble_x1y1_x2y2_norm[~slc_nan_rows]
    ensemble_scores = ensemble_scores[~slc_nan_rows]
    ensemble_labels = ensemble_labels[~slc_nan_rows]

    # Undo x1y1 x2y2 normalization
    ensemble_x1y1_x2y2 = ensemble_x1y1_x2y2_norm * np.tile(
        image_height_width, (1, 2)
    )

    # Add to dict with key = image_id
    detections_per_image_id[image_id] = {
        "boxes": ensemble_x1y1_x2y2,
        "scores": ensemble_scores,
        "labels": ensemble_labels,
    }

# %%
# Format as xarray dataset
ensemble_detections_ds = _detections_per_image_id_as_ds(
    detections_per_image_id
)

# %%
# Evaluate detections with hungarian

# ensemble_detections_ds = add_bboxes_min_max_corners(ensemble_detections_ds)

# add tp, fp, tp_iou as arrays to dataset?
# tp, fp, md, _ = evaluate_detections_hungarian(
#         ensemble_x1_y1_x2_y2, annots["boxes"], iou_threshold_precision
#     )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot ensemble detections on first image

# Get first image
image_index = 25
image = val_dataset[image_index][0]
gt_annotations = val_dataset[image_index][1]


plot_and_save_ensemble_detections(
    image=image,
    gt_boxes_x1_y1_x2_y2=gt_annotations["boxes"],
    pred_boxes_x1_y1_x2_y2=np.hstack(
        [
            ensemble_detections_ds[xy_corner_str]
            .isel(image_id=image_index)
            .values.T
            for xy_corner_str in ["xy_min", "xy_max"]
        ]
    ),
    pred_boxes_scores=ensemble_detections_ds.isel(
        image_id=image_index
    ).confidence.values,
    image_id=gt_annotations["image_id"],
    output_dir=Path.cwd(),
    precision=0.0,
    recall=0.0,
)

# %%
# Combine detections with WBF
# detections_ds = run_ensemble_of_detectors_on_dataset(
#     list_models,
#     dataset,  # could be list too
#     device,   # ensure models and dataset are placed on this device?
#     ensemble_boxes_method="wbf",
#     **ensemble_boxes_kwargs,
# )


# detections_ds = run_ensemble_of_detectors_on_dataloader(
#     list_models,
#     dataset,  # could be list too
#     device,   # ensure models and dataset are placed on this device?
#     ensemble_boxes_method="wbf",
#     **ensemble_boxes_kwargs,
# )
