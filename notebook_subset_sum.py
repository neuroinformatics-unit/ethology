"""Split an annotations dataset with an approximate subset sum algorithm."""

# %%

# %%
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

from ethology.detectors.datasets import (
    approximate_subset_sum,
    split_annotations_dataset_group_by,
)
from ethology.io.annotations import load_bboxes

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
sept2023_data_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full/")
aug2023_data_dir = Path("/home/sminano/swc/project_crabs/data/aug2023-full/")

sept2023_images_dir = sept2023_data_dir / "frames"
aug2023_images_dir = aug2023_data_dir / "frames"

sept2023_annots = (
    sept2023_data_dir / "annotations" / "VIA_JSON_combined_coco_gen.json"
)
aug2023_annots = (
    aug2023_data_dir / "annotations" / "VIA_JSON_combined_coco_gen.json"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read as a single annotation dataset

ds_all = load_bboxes.from_files(
    [sept2023_annots, aug2023_annots],
    format="COCO",
    images_dirs=[sept2023_images_dir, aug2023_images_dir],
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute video array and video dict and add to dataset


def split_at_any_delimiter(text: str, delimiters: list[str]) -> list[str]:
    """Split a string at any of the specified delimiters if present."""
    for delimiter in delimiters:
        if delimiter in text:
            return text.split(delimiter)
    return [text]


# Get video-pair filename per image
video_per_image_filename = np.array(
    [
        split_at_any_delimiter(
            ds_all.map_image_id_to_filename[i],
            # ["-Left", "-Right"],
            ["_frame"],
        )[0]
        for i in ds_all.image_id.values
    ]
)
print(video_per_image_filename.shape)


# Get list of unique video filenames
list_video_filenames = np.sort(np.unique(video_per_image_filename)).tolist()


# Assing IDs to unique video names as sorted alphabetically
map_video_filename_to_id = {
    str(k): i for i, k in enumerate(list_video_filenames)
}

# Add array to dataset
ds_all["video"] = xr.DataArray(
    np.array(
        [map_video_filename_to_id[vid] for vid in video_per_image_filename]
    ),
    dims="image_id",
)

# Add map to dataset
ds_all.attrs["map_video_id_to_filename"] = {
    v: k for k, v in map_video_filename_to_id.items()
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot frame count per video

map_video_to_frame_count = {
    str(k): len(list(g))
    for k, g in itertools.groupby(video_per_image_filename)
}

fig, ax = plt.subplots()

cmap = plt.cm.tab10
color_sept = cmap(0)
color_aug = cmap(1)
sept_bool = ["09.2023" in ky for ky in map_video_to_frame_count]
colors = [color_sept if sept else color_aug for sept in sept_bool]
ax.bar(
    map_video_to_frame_count.keys(),
    map_video_to_frame_count.values(),
    color=colors,
)
ax.set_xticks(range(len(map_video_to_frame_count)))
ax.set_xticklabels(map_video_to_frame_count.keys(), rotation=90, fontsize=8)
ax.set_ylabel("# frames")
ax.set_title("Frame count per video")

# Create custom legend entries
legend_elements = [
    Patch(facecolor=color_sept, label="09.2023"),
    Patch(facecolor=color_aug, label="08.2023"),
]
ax.legend(handles=legend_elements)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define dataset split

train_fraction = 0.8
test_fraction = 0.2

test_n_samples = int(test_fraction * len(ds_all.image_id))
train_n_samples = len(ds_all.image_id) - test_n_samples


print(f"train_n_samples: {train_n_samples}")
print(f"test_n_samples: {test_n_samples}")
print(f"total: {train_n_samples + test_n_samples}")
print(len(ds_all.image_id))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Split dataset into train and test sets with non-overlapping videos


# Compute test indices
seed = 42
tolerance = 0.0005

frame_count_per_video_id = {
    int(k): len(list(g)) for k, g in itertools.groupby(ds_all["video"].values)
}
test_idcs, test_total_frames = approximate_subset_sum(
    frame_count_per_video_id,
    test_n_samples,
    seed=seed,
    tolerance=tolerance,
)

print(f"Seed: {seed}")
for id in test_idcs:
    print(ds_all.map_video_id_to_filename[id])

print(f"Total sum: {test_total_frames}")
print(f"Target: {test_n_samples}")
print(f"Tolerance (samples): {test_n_samples * tolerance}")
print("------------------------------")

assert (
    sum(frame_count_per_video_id[id] for id in test_idcs) == test_total_frames
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create test and train dataset using computed indices


ds_test = ds_all.isel(image_id=ds_all["video"].isin(test_idcs))
ds_train = ds_all.isel(image_id=~ds_all["video"].isin(test_idcs))

assert all(
    np.unique(
        ds_all.isel(image_id=ds_all["video"].isin(test_idcs))["video"].values
    )
    == sorted(test_idcs)
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Alternatively

ds_test_2, ds_train_2 = split_annotations_dataset_group_by(
    ds_all,
    group_by_var="video",
    list_fractions=[train_fraction, test_fraction],
    seed=seed,
    tolerance=tolerance,
)

assert ds_test.equals(ds_test_2)
assert ds_train.equals(ds_train_2)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# To split by video-pair, define that array first

# Get video-pair filename per image
video_pair_per_image_filename = np.array(
    [
        split_at_any_delimiter(
            ds_all.map_image_id_to_filename[i],
            ["-Left", "-Right"],
        )[0]
        for i in ds_all.image_id.values
    ]
)
print(video_pair_per_image_filename.shape)

# Add array to dataset
ds_all["video_pair"] = xr.DataArray(
    video_pair_per_image_filename,
    dims="image_id",
)

print(len(np.unique(ds_all["video_pair"].values)))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Then split by video-pair

ds_test_video_pair, ds_train_video_pair = split_annotations_dataset_group_by(
    ds_all,
    group_by_var="video_pair",
    list_fractions=[train_fraction, test_fraction],
    seed=seed,
    tolerance=tolerance,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convert to torch datasets


# %%
# %%
# Randomly sample
# import numpy as np

# seed = 42
# rng = np.random.default_rng(seed)
# train_idcs = rng.choice(len(ds_all.image_id), train_n_samples, replace=False)
# test_idcs = rng.choice(len(ds_all.image_id), test_n_samples, replace=False)

# train_ds = ds_all.isel(image_id=train_idcs)
# test_ds = ds_all.isel(image_id=test_idcs)

# then convert to torch datasets?

# %%
