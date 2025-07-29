# %%
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from ensemble_boxes import weighted_boxes_fusion
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from ethology.datasets.create import create_coco_dataset
from ethology.detectors.inference import (
    collate_fn_varying_n_bboxes,
    concat_detections_ds,
    run_detector_on_dataloader,
    # run_detector_on_dataset,
)
from ethology.detectors.load import load_fasterrcnn_resnet50_fpn_v2
from ethology.detectors.utils import (
    add_bboxes_min_max_corners,
    detections_x1y1_x2y2_as_da_tuple,
    detections_x1y1_x2y2_as_ds,
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
# shuffle=False so that we dont shuffle the data after one pass over all batches
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
# Compute detections per model -- make it faster
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

# Fuse detections across models
# fused_detections_ds = combine_detections_across_models(all_models_detections_ds)

# Evaluate


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fuse detections across models -- Approach 1: naive
# v clear but slow

def wbf_wrapper(
    ds: xr.Dataset,  
) -> xr.Dataset:
    """Wrapper for weighted boxes fusion."""

    # Define parameters for WBF
    iou_thr_ensemble = 0.5
    skip_box_thr = 0.0001  # skip boxes with confidence below this threshold
    image_width_height = np.array([4096, 2160])

    # Check ds has required dimensions
    if "image_id" in ds.dims:
        raise ValueError("Input dataset must not have image_id dimension")
    if not all(s in ds.dims for s in ("model", "space", "id")):
        raise ValueError(
            "Input dataset must have model, space and id dimensions"
        )

    # Compute x1y1x2y2_normalised
    x1y1x2y2_normalised = xr.concat(
        [ds.xy_min, ds.xy_max], dim="space"
    ) / np.tile(image_width_height, (1, 2))[:,:,None]

    # Run WBF
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            x1y1x2y2_normalised.transpose("model", "id", "space"),
            ds.confidence,
            ds.label,
            iou_thr=iou_thr_ensemble,
            skip_box_thr=skip_box_thr,
        )
    )

    # Undo x1y1, x2y2 normalization
    ensemble_x1y1_x2y2 = ensemble_x1y1_x2y2_norm * np.tile(
        image_width_height, (1, 2)
    )

    # Format output as xarray dataarrays
    fused_detections_ds = detections_x1y1_x2y2_as_ds(
        ensemble_x1y1_x2y2, ensemble_scores, ensemble_labels
    )

    return fused_detections_ds


# %%
#%timeit --> 9.09 s ± 55.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# for every image_id slice
list_fused_detections_ds = []
for img_id in all_models_detections_ds.image_id:
    
    fused_detections_ds = wbf_wrapper(
        all_models_detections_ds.sel(image_id=img_id)
    )

    list_fused_detections_ds.append(fused_detections_ds)

# Concatenate fused detections across image_ids
fused_detections_ds = concat_detections_ds(
    list_fused_detections_ds,
    pd.Index(range(len(list_fused_detections_ds)), name="image_id"),
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fuse detections across models -- Approach 2: vectorized
# faster but less clear

def wbf_wrapper_arrays(
    x1y1: np.ndarray,  # model, annot, 4
    x2y2: np.ndarray,  # model, annot, 4
    confidence: np.ndarray,  # model, annot
    label: np.ndarray,  # model, annot
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:

    iou_thr_ensemble = 0.5
    skip_box_thr = 0.0001  # skip boxes with confidence below this threshold
    max_n_detections = 300  # set a priori, max after fusing
    image_width_height = np.array([4096, 2160])

    x1y1x2y2_normalised = (
        np.concat([x1y1, x2y2], axis=-1) / np.tile(image_width_height, (1, 2))
    )[:, :, :, None]

    # ------------------------------------
    # Run WBF
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            x1y1x2y2_normalised,
            confidence,
            label,
            iou_thr=iou_thr_ensemble,
            skip_box_thr=skip_box_thr,
        )
    )

    # ------------------------------------
    # Undo x1y1 x2y2 normalization
    ensemble_x1y1_x2y2 = ensemble_x1y1_x2y2_norm * np.tile(
        image_width_height, (1, 2)
    )

    # Combine x1y1, x2y2, scores and labels in one array
    ensemble_x1y2_x2y2_scores_labels = np.c_[
        ensemble_x1y1_x2y2, ensemble_scores, ensemble_labels
    ]

    # Remove rows with nan coordinates
    slc_nan_rows = np.any(np.isnan(ensemble_x1y1_x2y2), axis=1)
    ensemble_x1y2_x2y2_scores_labels = ensemble_x1y2_x2y2_scores_labels[
        ~slc_nan_rows
    ]

    # Pad combinedarray to max_n_detections
    # This is to have a constant output dimension in the `id` dimension
    ensemble_x1y2_x2y2_scores_labels = np.pad(
        ensemble_x1y2_x2y2_scores_labels,
        (
            (0, max_n_detections - ensemble_x1y2_x2y2_scores_labels.shape[0]),
            (0, 0),
        ),
        "constant",
        constant_values=np.nan,
    )

    # Format output as xarray dataarrays
    centroid, shape, confidence, label = detections_x1y1_x2y2_as_da_tuple(
        ensemble_x1y2_x2y2_scores_labels[:, 0:4],
        ensemble_x1y2_x2y2_scores_labels[:, 4],
        ensemble_x1y2_x2y2_scores_labels[:, 5],
    )

    # print(centroid.shape)  # space, id
    # print(shape.shape)  # space, id
    # print("<----->")

    return centroid, shape, confidence, label


# %%
# timeit --- 1.37 s ± 11.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# this will become a fn
centroid_fused, shape_fused, confidence_fused, label_fused = xr.apply_ufunc(
    wbf_wrapper_arrays,
    all_models_detections_ds.xy_min, # the underlaying .data array is passed
    all_models_detections_ds.xy_max,
    all_models_detections_ds.confidence,
    all_models_detections_ds.label,
    input_core_dims=[ # do not broadcast across these
        ["model", "id", "space"],  
        ["model", "id", "space"],  
        ["model", "id"],
        ["model", "id"],
    ],
    output_core_dims=[["space", "id"], ["space", "id"], ["id"], ["id"]],
    vectorize=True,
    # loop over non-core dims (i.e. image_id);
    # assumes function only takes arrays over core dims as input
    exclude_dims={"id"},
    # to allow dimensions that change size btw input and output
)


# Remove pad across annotations
centroid_fused = centroid_fused.dropna(dim="id", how="all")
shape_fused = shape_fused.dropna(dim="id", how="all")
confidence_fused = confidence_fused.dropna(dim="id", how="all")
label_fused = label_fused.dropna(dim="id", how="all")


# Pad labels with -1 rather than nan
label_fused = label_fused.fillna(-1)


# Return a dataset
fused_detections_ds = xr.Dataset(
    data_vars={
        "position": centroid_fused,
        "shape": shape_fused,
        "confidence": confidence_fused,
        "label": label_fused,
    }
)

# print(fused_detections_ds)

# %%


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

    # apply nms?
    # idcs_to_keep = torchvision.ops.nms(
    #     ensemble_x1y1_x2y2,
    #     ensemble_scores,
    #     iou_threshold=0.9,
    # )

    # ensemble_x1y1_x2y2 = ensemble_x1y1_x2y2[idcs_to_keep]
    # ensemble_scores = ensemble_scores[idcs_to_keep]
    # ensemble_labels = ensemble_labels[idcs_to_keep]

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
# # Combine detections with WBF
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


# %%

x1y1_x2y2_fused = xr.apply_ufunc(
    test,
    all_models_detections_ds.image_id,
    x1y1x2y2_norm.transpose(
        "model", "id", "space", "image_id"
    ),  # place broadcast dims at the end
    all_models_detections_ds.confidence.transpose("model", "id", "image_id"),
    all_models_detections_ds.label.transpose("model", "id", "image_id"),
    input_core_dims=[
        [],  # do not exclude any dimensions
        ["model", "id", "space"],  # do not broadcast across these
        ["model", "id"],
        ["model", "id"],
    ],
    output_core_dims=[["space", "id"]],
    vectorize=True,  # loop over non-core dims
    exclude_dims={
        "id"
    },  # to allow dimensions that change size btw input and output
)


print(x1y1_x2y2_fused.shape)  # image_id, 4, padded_id

# Can I remove the excessive pad?
