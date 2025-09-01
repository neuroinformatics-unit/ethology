# %%
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
import xarray as xr
from torch import vmap
from torch.func import functional_call, stack_module_state
from torch.utils.data import random_split

from ethology.datasets.create import create_coco_dataset
from ethology.detectors.load import load_fasterrcnn_resnet50_fpn_v2
from ethology.mlflow import (
    read_cli_args_from_mlflow_params,
    read_config_from_mlflow_params,
    read_mlflow_params,
)

# Set xarray options
xr.set_options(display_expand_attrs=False)


# %%
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
    # "above_1st": ml_runs_experiment_dir / "879d2f77e2b24adcb06b87d2fede6a04",
    # "above_5th": ml_runs_experiment_dir / "75583ec227e3444ab692b99c64795325",
    # "above_10th": ml_runs_experiment_dir / "4acc37206b1e4f679d535c837bee2c2f",
    "above_25th": ml_runs_experiment_dir / "fdcf88fcbcc84fbeb94b45ca6b6f8914",
    "above_50th": ml_runs_experiment_dir / "daa05ded0ea047388c9134bf044061c5",
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


# %%
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


# %%
# Check that all models have the same dataset config
ref_config = list_config[0]
for key in ["train_fraction", "val_over_test_fraction"]:
    assert all(config[key] == ref_config[key] for config in list_config)

ref_cli_args = list_cli_args[0]
assert all(
    cli_args["seed_n"] == ref_cli_args["seed_n"] for cli_args in list_cli_args
)

# %%
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


# %%
# Check output of a single model

# model = list_models[0]
# model.to(device)

# img, annot = val_dataset[0]

# with torch.no_grad():
#     detections = model(img.to(device)[None])

# a list with one dict per element in the batch, each with keys:
# - boxes: tensor of shape [N, 4]
# - scores: tensor of shape [N]
# - labels: tensor of shape [N]


# %%
# Naive

# predictions_val_set_per_model = []
# with torch.no_grad():
#     for model in list_models:
#         model.to(device)
#         predictions_val_set_per_model.append(
#             [model(img.to(device)[None])[0] for img, _annot in val_dataset_subset]
#         )
# # [None]  # [1, C, H, W] -- add batch dimension

# %%
# Vectorized

# set a max number of detections per image
max_detections_per_image = 200

# stack params and buffers across models
# Given a list of M nn.Modules of the same class,
# returns two dictionaries that stack all of their parameters
# and buffers together, indexed by name
params, buffers = stack_module_state(list_models)


# define fn for vmap
single_meta_model = copy.deepcopy(list_models[0]).to("meta")


def wrapper_model(params, buffers, img):
    # Performs a functional call on the module by replacing the
    # module parameters and buffers with the provided ones.
    # Returns the result of calling `single_meta_model`
    list_detection_dicts = functional_call(
        single_meta_model,
        (params, buffers),
        (img,),  # [B, C, H, W]
        strict=True,
    )  # one dict per element in the batch

    # pad to 200 detections per image
    list_detection_dicts_padded = []
    for detection_dict in list_detection_dicts:
        n_detections = detection_dict["boxes"].shape[0]
        detection_dict_padded = {}

        detection_dict_padded["boxes"] = F.pad(
            detection_dict["boxes"],
            (0, 0, 0, max_detections_per_image - n_detections),
            mode="constant",
            value=np.nan,
        )
        detection_dict_padded["scores"] = F.pad(
            detection_dict["scores"],
            (0, max_detections_per_image - n_detections),
            mode="constant",
            value=np.nan,
        )
        detection_dict_padded["labels"] = F.pad(
            detection_dict["labels"],
            (0, max_detections_per_image - n_detections),
            mode="constant",
            value=-1,
        )

        list_detection_dicts_padded.append(detection_dict_padded)

    return list_detection_dicts_padded


# %%
# Run wrapper function on single model

model = list_models[0]

model.eval()  #

params_one_model = dict(model.named_parameters())
buffers_one_model = dict(model.named_buffers())

# place on device
params_one_model = {k: v.to(device) for k, v in params_one_model.items()}
buffers_one_model = {k: v.to(device) for k, v in buffers_one_model.items()}

# get data
val_dataset_subset = torch.utils.data.Subset(val_dataset, range(1))
val_dataset_images = torch.stack(
    [img.to(device) for img, _annot in val_dataset_subset]
)

# %%
out = wrapper_model(params_one_model, buffers_one_model, val_dataset_images)
# %%


# %%
# place params and buffers on device
# (rather than models  + params and buffers)
params = {k: v.to(device) for k, v in params.items()}
buffers = {k: v.to(device) for k, v in buffers.items()}


# prepare data for vmap
val_dataset_subset = torch.utils.data.Subset(val_dataset, range(1))

val_dataset_images = torch.stack(
    [img.to(device) for img, _annot in val_dataset_subset]
)


# %%

# compute predictions using vmap
# in_dims Specifies which dimension of the inputs to `fmodel` should be mapped over.
predictions_val_set_per_model_vmap = vmap(wrapper_model, in_dims=(0, 0, None))(
    params,
    buffers,
    val_dataset_images,
)


# %%
