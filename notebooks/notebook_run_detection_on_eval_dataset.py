"""Run detection on a Pytorch dataset and export results as a movement dataset.

A script to run detection only (no tracking) on a Pytorch dataset and
export the results in a format that can be loaded in movement napari widget.
"""

# %%
import json
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from torch.utils.data import random_split

from ethology.datasets.create import create_coco_dataset
from ethology.detectors.inference import run_detector_on_dataset
from ethology.detectors.load import load_fasterrcnn_resnet50_fpn_v2
from ethology.mlflow import (
    read_cli_args_from_mlflow_params,
    read_config_from_mlflow_params,
    read_mlflow_params,
)

# Set xarray options
xr.set_options(display_expand_attrs=False)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data - in domain
dataset_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full")
annotations_dir = Path("/home/sminano/swc/project_ethology/large_annotations")

experiment_ID = "617393114420881798"
ml_runs_experiment_dir = (
    Path("/home/sminano/swc/project_crabs/ml-runs") / experiment_ID
)

# I pick seed 42 for each set of models
models_dict = {
    # "above_0th": ml_runs_experiment_dir / "f348d9d196934073bece1b877cbc4d38",
    "above_1st": ml_runs_experiment_dir / "879d2f77e2b24adcb06b87d2fede6a04",
    # "above_5th": ml_runs_experiment_dir / "75583ec227e3444ab692b99c64795325",
    # "above_10th": ml_runs_experiment_dir / "4acc37206b1e4f679d535c837bee2c2f",
    # "above_25th": ml_runs_experiment_dir / "fdcf88fcbcc84fbeb94b45ca6b6f8914",
    # "above_50th": ml_runs_experiment_dir / "daa05ded0ea047388c9134bf044061c5",
}

output_dir = Path(
    "/home/sminano/swc/project_ethology/remove_small_bboxes_inD_output"
)
# create output dir if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute detections for each model

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        device=device,
    )
    model.eval()

    # ------------------------------------
    # Define transforms for inference
    inference_transforms = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    # Create COCO dataset
    annotations_filename = Path(cli_args["annotation_files"][0]).name
    print(annotations_filename)
    print(f"Seed: {cli_args['seed_n']}")

    dataset_coco = create_coco_dataset(
        images_dir=Path(dataset_dir) / "frames",
        annotations_file=annotations_dir / annotations_filename,
        composed_transform=inference_transforms,
    )

    # Split dataset like in crabs repo
    train_dataset, val_dataset, test_dataset = split_dataset_crab_repo(
        dataset_coco,
        seed_n=cli_args["seed_n"],
        config=config,
    )

    # ------------------------------------

    # Run detection on validation dataset
    detections_ds = run_detector_on_dataset(
        model=model,
        dataset=val_dataset,
        device=device,
    )

    # ------------------------------------
    # Add attributes to detections dataset
    detections_ds.attrs["model_name"] = "fasterrcnn_resnet50_fpn_v2"
    detections_ds.attrs["model_path"] = trained_model_path
    detections_ds.attrs["config"] = json.dumps(config, indent=2)
    detections_ds.attrs["cli_args"] = json.dumps(cli_args, indent=2)
    detections_ds.attrs["dataset_dir"] = str(dataset_dir)
    detections_ds.attrs["annotations_file"] = str(
        annotations_dir / annotations_filename
    )
    detections_ds.attrs["seed_n"] = cli_args["seed_n"]
    detections_ds.attrs["coco_crabs_dataset_split"] = "val"

    # ------------------------------------
    # # Save detections dataset
    # detections_ds.to_netcdf(
    #     output_dir
    #     / f"{model_key}_detections_val_set_seed_{cli_args['seed_n']}_{timestamp}.nc"
    # )



# %%%%%%%%%%%%%%%%%%%%%%%
# %%time
# Use dataloader to run detection on validation set
# val_dataloader = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=8,
#     shuffle=False,
#     collate_fn=collate_fn_varying_n_bboxes,
# )

# # Run detection on dataloader
# detections_dict_per_batch = run_detector_on_dataloader(
#     model=model,
#     dataloader=val_dataloader,
#     device=device,
# )
