# %%
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

from ethology.io.annotations import load_bboxes
from ethology.torch_dataset import from_annotations_dataset

# %%
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

# %%
# Read as a single annotation dataset

# ds_sept = load_bboxes.from_files(
#     sept2023_annots, format="COCO", images_dirs=sept2023_images_dir
# )
# ds_aug = load_bboxes.from_files(
#     aug2023_annots, format="COCO", images_dirs=aug2023_images_dir
# )

# xr.concat only keeps attributes from the first dataset:
# ds_all = xr.concat([ds_sept, ds_aug], dim="image_id")

ds_all = load_bboxes.from_files(
    [sept2023_annots, aug2023_annots],
    format="COCO",
    images_dirs=[sept2023_images_dir, aug2023_images_dir],
)


# %%
# Compute video_array and add to dataset

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

list_video_filenames = np.sort(np.unique(video_per_image_filename)).tolist()

# %%
# Assing IDs to unique video as sorted alphabetically
map_video_filename_to_id = {
    str(k): i
    for i, k in enumerate(list_video_filenames)
}


# Add video-pair ID to dataset and map to filename
# as an attribute
ds_all["video"] = xr.DataArray(
    np.array(
        [map_video_filename_to_id[vid] for vid in video_per_image_filename]
    ),
    dims="image_id",
)
ds_all.attrs["map_video_id_to_filename"] = {
    v: k for k, v in map_video_filename_to_id.items()
}


# %%
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

# video_array = np.unique(video_per_image_filename)
# print(video_array)
# print(video_array.shape)

# %%%%%%%%%%%
# Subset sum: compute all subset indices that add up to target


def backtrack(
    curr_index,
    current_subset_idcs,
    target,
    list_values,
    all_solution_subsets,
):
    """Node operation.

    Three types of nodes: solution, dead or branching node.

    Written as pure function.
    TODO: apparently this would be more efficient if the values are sorted

    curr_index:
        Index to consider in current node
    current_subset_idcs:
        Indices for the values included in the running subset
    target:
        Sum the subset needs to add to.
    list_values:
        superset of numbers to extract subsets from
    all_solutions_subsets:
        a list with all subsets of indices that sum up to the target

    """
    current_sum = sum([list_values[idx] for idx in current_subset_idcs])
    remaining_sum = target - current_sum  # ok?
    n_values = len(list_values)

    # Solution node
    if remaining_sum == 0:
        all_solution_subsets.append(tuple(current_subset_idcs))
        return all_solution_subsets

    # Dead node
    if remaining_sum < 0 or curr_index >= n_values:
        return all_solution_subsets

    # Branching node
    # Branch 1: include current index and move to next node
    current_subset_idcs.append(curr_index)
    all_solution_subsets = backtrack(
        curr_index=curr_index + 1,
        current_subset_idcs=current_subset_idcs,
        target=target,
        list_values=list_values,
        all_solution_subsets=all_solution_subsets,
    )

    # Branch 2: exclude branch and move to next node
    current_subset_idcs.pop()  # remove last element
    all_solution_subsets = backtrack(
        curr_index=curr_index + 1,
        current_subset_idcs=current_subset_idcs,
        target=target,
        list_values=list_values,
        all_solution_subsets=all_solution_subsets,
    )

    return all_solution_subsets


# %%
def compute_all_idcs_sum_to_target(list_values, target):
    """Subset sum indices.

    Given a list of values and a target sum, compute all the subsets of indices that add up to the target.

    Parameters
    ----------
    list_values : list
        The list of values to consider. They should be strictly positive
        integers.
    target : int
        The target sum to achieve.

    Returns
    -------
    all_sol_indices : list[list[int]]
        The indices of the subsets that add up to the target.

    """
    if not all(isinstance(x, int) and x > 0 for x in list_values):
        raise ValueError("list_values should be a list of all integers > 0")

    if target > sum(list_values):
        raise ValueError("target is greater than the sum of the list_values")

    if not isinstance(target, int):
        raise ValueError("target should be an integer")

    all_sol_indices = backtrack(
        curr_index=0,
        current_subset_idcs=[],
        target=target,
        list_values=list_values,
        all_solution_subsets=[],
    )

    return all_sol_indices


# %%
# Simple example

list_values = [2, 4, 6, 7, 5, 5, 3, 9, 1]
target = 10
# FIX: depends on the order of the list_values?
# if O is at start or end, the solution is different -- I exclude 0 now
all_sol_indices = compute_all_idcs_sum_to_target(list_values, target=target)

print(all_sol_indices)
print(len(all_sol_indices))
print("----")
for elem in all_sol_indices:
    print(f"idcs: {elem}")
    print(f"values: {[list_values[x] for x in elem]}")
    print(f"sum: {sum(list_values[x] for x in elem)}")
    print("----")


# %%
# Check implementation against brute force: itertools combinations
# (for testing)
# we compare the values, not the indices


def compute_all_subsets_sum_to_target_brute_force(list_values, target):
    """Compute all subsets that sum up to the target value.

    Parameters
    ----------
    list_values : list
        The list of values to consider. They should be strictly positive
        integers.
    target : int
        The target sum to achieve.

    Returns
    -------
    all_subsets_per_length : dict[int, np.ndarray]
        A dictionary with the length of the subsets as keys and the subsets as
        values. The M subsets of length N are represented as numpy arrays of
        shape (M, N).

    """
    if not all(isinstance(x, int) and x > 0 for x in list_values):
        raise ValueError("list_values should be a list of all integers > 0")

    # brute force
    all_subsets_per_length = {}
    for n in range(len(list_values)):
        # Get all subsets of that length
        all_combinations_length_n = np.array(
            list(itertools.combinations(list_values, n))
        )  # K x N

        # Get subsets that sum up to the target
        idcs_with_target_sum = np.where(
            all_combinations_length_n.sum(axis=1) == target
        )[0]

        all_subsets_per_length[n] = [
            all_combinations_length_n[idx, :] for idx in idcs_with_target_sum
        ]

    return all_subsets_per_length


# run brute force
all_subsets_per_length = compute_all_subsets_sum_to_target_brute_force(
    list_values, target
)

# express as a list of lists
all_sol_values_brute_force = []
for val in all_subsets_per_length.values():
    all_sol_values_brute_force.extend(sorted(elem.tolist()) for elem in val)

# sort the values for comparison with brute force
all_sol_values = [
    sorted([list_values[x] for x in idcs]) for idcs in all_sol_indices
]

assert sorted(all_sol_values) == sorted(all_sol_values_brute_force)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Split dataset into train and test sets with non-overlapping videos

train_fraction = 0.8
test_fraction = 0.2

train_n_samples = int(train_fraction * len(ds_all.image_id))
test_n_samples = len(ds_all.image_id) - train_n_samples


print(f"train_n_samples: {train_n_samples}")
print(f"test_n_samples: {test_n_samples}")
print(f"total: {train_n_samples + test_n_samples}")
print(len(ds_all.image_id))
# %%

frame_count_per_video_id = {
    int(k): len(list(g)) for k, g in itertools.groupby(ds_all["video"].values)
}

# sort by key (i.e. video ID)
frame_count_per_video_id = dict(sorted(frame_count_per_video_id.items()))

# returns indices of the array that sum to the target
# choose the smallest one
all_test_videos_idcs = compute_all_idcs_sum_to_target(
    list(frame_count_per_video_id.values()), test_n_samples
)  # 12M options!


# train_videos_idcs = compute_all_idcs_sum_to_target(
#     list(frame_count_per_video_id.values()), train_n_samples
# )


# test_videos_idcs = set(range(len(frame_count_per_video_id))) - set(
#     train_videos_idcs
# )

# log: one of N solutions


# %%
# Get first combination with Aug video
comb = next(
    idcs
    for idcs in all_test_videos_idcs
    for x in [ds_all.map_video_id_to_filename[id] for id in idcs]
    if "08.2023" in x
)


# %%
# write to file
import pickle

with open("all_test_videos_idcs.pkl", "wb") as f:
    pickle.dump(all_test_videos_idcs, f)


# %%
# Very slow.... maybe
# - check if possible
# - if not, get an approximate


def subset_sum_dp_boolean(arr: list[int], target: int) -> bool:
    """
    Dynamic programming approach - returns True if ANY subset sums to target.
    Time complexity: O(n * target), Space complexity: O(target)
    """
    if target < 0:
        return False

    dp = [False] * (target + 1)
    dp[0] = True

    for num in arr:
        # Iterate backwards to avoid using the same element twice
        for i in range(target, num - 1, -1):
            if dp[i - num]:
                dp[i] = True

    return dp[target]


# def subset_sum_dp_with_path(
#     arr: list[int], target: int
# ) -> list[tuple[int, ...]]:
#     """
#     Dynamic programming approach that finds ALL subsets that sum to target.
#     More memory intensive but finds all solutions.
#     Time complexity: O(n * target * 2^n) in worst case, Space complexity: O(target * 2^n)
#     """
#     if target < 0:
#         return []

#     n = len(arr)
#     # dp[i] will contain all possible subsets that sum to i
#     dp = [[] for _ in range(target + 1)]
#     dp[0] = [()]  # Empty subset sums to 0

#     for i, num in enumerate(arr):
#         if num > target:
#             continue

#         # Iterate backwards to avoid using the same element twice
#         for j in range(target, num - 1, -1):
#             if dp[j - num]:
#                 for subset in dp[j - num]:
#                     # Add current element to all subsets that sum to j-num
#                     new_subset = subset + (i,)
#                     dp[j].append(new_subset)

#     return dp[target]


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
