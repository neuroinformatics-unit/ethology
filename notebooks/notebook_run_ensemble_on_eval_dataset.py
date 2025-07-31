# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
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
from ethology.detectors.ensembles import (
    combine_detections_across_models_wbf,
    wbf_wrapper_arrays,
)
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
    detections_x1y1_x2y2_as_da_tuple,
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

dataset_dir = Path("/home/sminano/swc/project_crabs/data/aug2023-full")
# Path("/home/sminano/swc/project_crabs/data/sep2023-full")
annotations_dir = Path(
    "/home/sminano/swc/project_crabs/data/aug2023-full/annotations"
)
annotations_file_path = annotations_dir / "VIA_JSON_combined_coco_gen.json"

experiment_ID = "617393114420881798"
ml_runs_experiment_dir = (
    Path("/home/sminano/swc/project_crabs/ml-runs") / experiment_ID
)

# I pick seed 42 for each set of models
models_dict = {
    "above_0th": ml_runs_experiment_dir / "f348d9d196934073bece1b877cbc4d38",
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
    extra_str="",
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
    extra_str = f"{extra_str}_" if extra_str else ""
    output_filename = output_dir / f"val_set_{extra_str}{image_id:06d}.png"
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

    print(
        f"Run name: {mlflow_params['run_name']}, trained on "
        f"{Path(cli_args['dataset_dirs'][0]).name}, "
        f"{Path(cli_args['annotation_files'][0]).name}"
    )
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
    # config=ref_config,
    config={
        "train_fraction": 0.0,
        "val_over_test_fraction": 1.0,
    },  # only uses train_fraction and val_over_test_fraction
)

print(annotations_file_path)

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
fused_detections_ds = combine_detections_across_models_wbf(  # ------soft_nms?
    all_models_detections_ds,
    kwargs_wbf={
        "iou_thr_ensemble": 0.5,
        "skip_box_thr": 0.0001,
        "max_n_detections": 300,  # set default?
    },
)

# %%
from ensemble_boxes import weighted_boxes_fusion


def wbf_wrapper_arrays_2(
    bboxes_x1y1: np.ndarray,
    bboxes_x2y2: np.ndarray,  # model, annot, 4
    confidence: np.ndarray,  # model, annot
    label: np.ndarray,  # model, annot
    image_width_height: np.ndarray,  # = np.array([4096, 2160]),
    iou_thr_ensemble: float = 0.5,
    skip_box_thr: float = 0.0001,
    max_n_detections: int = 300,  # should be larger than the max number of detections fused per image
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Wrap weighted boxes fusion to receive arrays as input.

    Parameters
    ----------
    bboxes_x1y1: np.ndarray
        Detected bounding boxes in a single imagein x1y1 format, with shape
        n_models, n_annotations, 2.
    bboxes_x2y2: np.ndarray
        Detected bounding boxes in a single image in x2y2 format, with shape
        n_models, n_annotations, 2.
    confidence: np.ndarray
        Confidence scores for each bounding box, with shape
        n_models, n_annotations.
    label: np.ndarray
        Labels for each bounding box, with shape n_models, n_annotations.
    image_width_height: np.ndarray
        Width and height of the image, with shape 2.
    iou_thr_ensemble: float
        IoU threshold for detections to be considered for fusion.
    skip_box_thr: float
        Threshold for skipping boxes with confidence below this value.
    max_n_detections: int
        Fused bounding boxes arrays are padded to this total number of boxes.
        Its value should be larger than the expected maximum number of detections
        per image after fusing across models.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        Tuple of xr.DataArrays containing the fused detections. The arrays
        are padded to max_n_detections and contain the data for the centroid,
        shape, confidence and label of the fused detections.

    """
    # Prepare bboxes for WBF
    bboxes_x1y1_x2y2_normalised = np.concat(
        [bboxes_x1y1, bboxes_x2y2], axis=-1
    ) / np.tile(image_width_height, (1, 2))  # [:, :, :, None]

    # ------------------------------------
    # Run WBF
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            bboxes_x1y1_x2y2_normalised,
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

    # Pad combined array to max_n_detections
    # (this is required to concatenate across image_ids
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

    return centroid, shape, confidence, label


# %%
from ensemble_boxes import soft_nms


def soft_nms_wrapper_arrays_2(
    bboxes_x1y1: np.ndarray,
    bboxes_x2y2: np.ndarray,  # model, annot, 4
    confidence: np.ndarray,  # model, annot
    label: np.ndarray,  # model, annot
    image_width_height: np.ndarray,  # = np.array([4096, 2160]),
    iou_thr_ensemble: float = 0.5,
    skip_box_thr: float = 0.0001,
    max_n_detections: int = 300,  # should be larger than the max number of detections fused per image
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Wrap weighted boxes fusion to receive arrays as input.

    Parameters
    ----------
    bboxes_x1y1: np.ndarray
        Detected bounding boxes in a single imagein x1y1 format, with shape
        n_models, n_annotations, 2.
    bboxes_x2y2: np.ndarray
        Detected bounding boxes in a single image in x2y2 format, with shape
        n_models, n_annotations, 2.
    confidence: np.ndarray
        Confidence scores for each bounding box, with shape
        n_models, n_annotations.
    label: np.ndarray
        Labels for each bounding box, with shape n_models, n_annotations.
    image_width_height: np.ndarray
        Width and height of the image, with shape 2.
    iou_thr_ensemble: float
        IoU threshold for detections to be considered for fusion.
    skip_box_thr: float
        Threshold for skipping boxes with confidence below this value.
    max_n_detections: int
        Fused bounding boxes arrays are padded to this total number of boxes.
        Its value should be larger than the expected maximum number of detections
        per image after fusing across models.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        Tuple of xr.DataArrays containing the fused detections. The arrays
        are padded to max_n_detections and contain the data for the centroid,
        shape, confidence and label of the fused detections.

    """
    # Prepare bboxes for WBF
    bboxes_x1y1_x2y2_normalised = np.concat(
        [bboxes_x1y1, bboxes_x2y2], axis=-1
    ) / np.tile(image_width_height, (1, 2))  # [:, :, :, None]

    # Remove rows with nan coordinates
    n_models = bboxes_x1y1_x2y2_normalised.shape[0]
    list_bboxes_per_model = [
        arr.squeeze() for arr in np.split(
            bboxes_x1y1_x2y2_normalised, n_models, axis=0
        )
    ]
    list_bboxes_per_model = [
        arr[~np.any(np.isnan(arr), axis=1), :]
        for arr in list_bboxes_per_model
    ]
    list_confidence_per_model = [
        conf_arr.squeeze()[:bbox_arr.shape[0]]
        for bbox_arr, conf_arr in zip(
            list_bboxes_per_model,
            np.split(confidence, n_models, axis=0),
            strict=True,
        )
    ]
    list_label_per_model = [
        label_arr.squeeze()[:bbox_arr.shape[0]]
        for bbox_arr, label_arr in zip(
            list_bboxes_per_model,
            np.split(label, n_models, axis=0),
            strict=True,
        )
    ]
    # list_label_per_model = [
    #     label_arr[bbox_arr.shape[0], :]
    #     for bbox_arr, label_arr in zip(
    #         list_bboxes_per_model, np.split(label, n_models, axis=0)
    #     )
    # ]

    # ------------------------------------
    # Run WBF
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = soft_nms(
        #bboxes_x1y1_x2y2_normalised,
        list_bboxes_per_model,
        list_confidence_per_model,
        list_label_per_model,
        iou_thr=iou_thr_ensemble,
        thresh=skip_box_thr,  # threshold for boxes to keep
        method=3,  # 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
        sigma=0.5,  # sigma for gaussian soft-NMS
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

    # Pad combined array to max_n_detections
    # (this is required to concatenate across image_ids
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

    return centroid, shape, confidence, label


# %%
# Prepare kwargs
kwargs_wbf = {
    "iou_thr_ensemble": 0.5,
    "skip_box_thr": 0.0001,
    "max_n_detections": 500,  # set default?
}
kwargs_wbf["image_width_height"] = np.array(
    [
        all_models_detections_ds.attrs[img_size]
        for img_size in ["image_width", "image_height"]
    ]
)

# Run WBF vectorized
centroid_fused, shape_fused, confidence_fused, label_fused = xr.apply_ufunc(
    soft_nms_wrapper_arrays_2,  # ------------#wbf_wrapper_arrays,
    all_models_detections_ds.xy_min,  # the underlaying .data array is passed
    all_models_detections_ds.xy_max,
    all_models_detections_ds.confidence,
    all_models_detections_ds.label,
    kwargs=kwargs_wbf,
    input_core_dims=[  # do not broadcast across these
        ["model", "id", "space"],
        ["model", "id", "space"],
        ["model", "id"],
        ["model", "id"],
    ],
    output_core_dims=[
        ["space", "id"],
        ["space", "id"],
        ["id"],
        ["id"],
    ],
    vectorize=True,
    # loop over non-core dims (i.e. image_id);
    # assumes function only takes arrays over core dims as input
    exclude_dims={"id"},
    # to allow dimensions that change size btw input and output
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define ground truth dataset

# read annotations as a dataset
print(annotations_file_path)
gt_bboxes_ds = load_bboxes.from_files(annotations_file_path, format="COCO")

# fix category ID (to be fixed in loader)
gt_bboxes_ds["category"] = gt_bboxes_ds["category"].where(
    gt_bboxes_ds["category"] != 0, 1
)

# select only image_id in val_dataset
# Note that the max number of annotations per image in the val_dataset
# will stay as in the original dataset (also, category = -1 is not considered
# an empty value for xarrays .dropna())
list_image_ids_val = [annot["image_id"] for img, annot in val_dataset]
gt_bboxes_val_ds = gt_bboxes_ds.sel(image_id=list_image_ids_val)


# %%
# Alternatively: convert torch dataset into xarray detections dataset
# .....
# is it faster?
val_ds = torch_dataset_to_xr_dataset(val_dataset)

# # %%
# # check data arrays are the same but with annotations in different order
# # There is no guarantee that annotation with id=15 is the same in the
# # xr dataset computed from the annotations file and the one computed from
# # the torch dataset.
# for idx in range(len(val_ds.image_id.values)):
#     idcs_sorted_x1 = np.lexsort(
#         (
#             gt_bboxes_val_ds.position.values[idx, 1, :],
#             gt_bboxes_val_ds.position.values[idx, 0, :],
#         )
#     )  # sort by x, then y
#     idcs_sorted_x2 = np.lexsort(
#         (
#             val_ds.position.values[idx, 1, :],
#             val_ds.position.values[idx, 0, :],
#         )
#     )  # sort by x, then y
#     assert np.allclose(
#         gt_bboxes_val_ds.position.values[idx, :, idcs_sorted_x1],
#         val_ds.position.values[idx, :, idcs_sorted_x2],
#         equal_nan=True,
#         rtol=1e-5,
#         atol=1e-8,
#     )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate

# ensemble model
fused_detections_ds, gt_bboxes_val_ds = compute_precision_recall_ds(
    pred_bboxes_ds=fused_detections_ds,
    gt_bboxes_ds=gt_bboxes_val_ds,
    iou_threshold=0.1,  # change to 0.5?
)

print(f"Precision: {fused_detections_ds.precision.mean().values:.4f}")
print(f"Recall: {fused_detections_ds.recall.mean().values:.4f}")

# %%
# single models
list_detections_ds_eval = []
for k, ds in enumerate(list_detections_ds):
    detections_ds, _ = compute_precision_recall_ds(
        pred_bboxes_ds=ds,
        gt_bboxes_ds=gt_bboxes_val_ds,
        iou_threshold=0.1,  # change to 0.5?
    )
    list_detections_ds_eval.append(detections_ds)

    print(f"Model: {k}")
    print(f"Precision: {detections_ds.precision.mean().values:.4f}")
    print(f"Recall: {detections_ds.recall.mean().values:.4f}")
    print("--------------------------------")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot ensemble detections on a selected image

# idcs_low_precision = np.argwhere(fused_detections_ds.precision.data < 0.5)
# idcs_high_precision = np.argwhere(fused_detections_ds.precision.data > 0.9)

fused_detections_ds = detections_ds
idcs_imgs_increasing_precision = np.argsort(fused_detections_ds.precision.data)
step = 5  # 50


# Get first image
for i in list(range(0, len(idcs_imgs_increasing_precision), step)) + [
    len(idcs_imgs_increasing_precision) - 1
]:
    image_index = idcs_imgs_increasing_precision[i].item()
    image, gt_annotations = val_dataset[image_index]

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
        output_dir=Path(
            "/home/sminano/swc/project_ethology/0th-percentile-ood-aug2023"
        ),
        extra_str=f"{i:03d}",
        precision=fused_detections_ds.isel(
            image_id=image_index
        ).precision.values,
        recall=fused_detections_ds.isel(image_id=image_index).recall.values,
    )
    print(f"image id: {gt_annotations['image_id']}")
# %%
# %matplotlib widget
# %%
fig, ax = plt.subplots()
ax.hist(fused_detections_ds.precision.values)
ax.axvline(
    fused_detections_ds.precision.values.mean(), color="red", linestyle="--"
)
ax.set_xlim(0, 1)
ax.set_xlabel("Precision per frame")
ax.set_ylabel("count (frames)")
ax.set_title(f"Precision OOD (n={fused_detections_ds.sizes['image_id']})")

fig, ax = plt.subplots()
ax.hist(fused_detections_ds.recall.values)
ax.axvline(
    fused_detections_ds.recall.values.mean(), color="red", linestyle="--"
)
ax.set_xlim(0, 1)
ax.set_xlabel("Recall per frame")
ax.set_ylabel("count (frames)")
ax.set_title(f"Recall OOD (n={fused_detections_ds.sizes['image_id']})")

# plt.show()
# %%
