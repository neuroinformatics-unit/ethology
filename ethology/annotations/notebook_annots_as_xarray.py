"""Explore formatting COCO annotations as an xarray Dataset.

The dataset is made up from the following data variables:
- bbox:  a 3D array with bounding box coordinates and shape
        (max_n_bboxes_per_image, n_images, 4).
        The four coordinates represent (x, y, h, w) per annotation.
- global_id: a 2D array of shape (max_n_bboxes_per_image, n_images) with
        the global ID of each annotation.

To add:
- category: a 2D array of shape (max_n_bboxes_per_image, n_images) with
        the category ID / str of each annotation.
- split bbox into position and shape.
- keep track of image filename?

"""

# %%%%%%%%%%%%%%%%%%%%
# imports

import numpy as np
import xarray as xr
from utils import read_json_file_as_dict

# %%%%%%%%%%%%%%%%%%%
# input data
via_file_path = (
    "/home/sminano/swc/project_ethology/sample_VIA_annotations/VIA_JSON_1.json"
)
coco_file_path = (
    "/home/sminano/swc/project_ethology/sample_COCO_annotations/sample_annotations_1.json"
)

# via_data = read_via_json_file_as_dict(via_file_path)
# print(via_data.keys())  # _via_img_metadata, _via_image_id_list

# %%%%%%%%%%%%%%%%%%%%
# read input json as dict
coco_data = read_json_file_as_dict(coco_file_path)

print(
    coco_data.keys()
)  # dict_keys(['annotations', 'categories', 'images', 'info', 'licenses'])


# %%%%%%%%%%%%%%%%%%%%
# helper fn to format data as homogeneous arrays
def compute_homog_data_array_per_image_id(data_str, axis_image_id_in_output):
    # pair up data with image id
    pair_data = []
    for annot in coco_data["annotations"]:
        if isinstance(annot[data_str], list):
            pair_data.append(annot[data_str] + [annot["image_id"]])
        else:
            pair_data.append([annot[data_str], annot["image_id"]])

    data_and_image_id_array = np.array(pair_data)

    # split
    data_array_per_image_id = np.split(
        data_and_image_id_array[:, : data_and_image_id_array.shape[1] - 1],
        np.where(np.diff(data_and_image_id_array[:, -1]))[0] + 1,
        axis=0,
    )

    # pad missing annotation-image IDs
    max_bboxes_per_image = max([d.shape[0] for d in data_array_per_image_id])
    data_array_per_image_id_with_nans = np.stack(
        [
            np.concat(
                (
                    d,
                    np.full(
                        (max_bboxes_per_image - d.shape[0], d.shape[1]), np.nan
                    ),
                )
            ).squeeze()
            for d in data_array_per_image_id
        ],
        axis=axis_image_id_in_output,  # 1, -1
    )  # annotation_image_id, image_id, space

    return data_array_per_image_id_with_nans


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Format data

# define bboxes data array
bboxes_data = compute_homog_data_array_per_image_id(
    "bbox", axis_image_id_in_output=1
)

# define annot ID data array
annot_ID_data = compute_homog_data_array_per_image_id(
    "id", axis_image_id_in_output=-1
)

# %%%%%%%%%%%%%%%%%%%%
# Create xarray Dataset
ds = xr.Dataset(
    data_vars=dict(
        bbox=(["annotation_image_id", "image_id", "space"], bboxes_data),
        global_id=(
            ["annotation_image_id", "image_id"],
            annot_ID_data,
        ),
    ),
    coords=dict(
        annotation_image_id=list(range(bboxes_data.shape[0])),
        image_id=np.unique(
            [annot["image_id"] for annot in coco_data["annotations"]]
        ),
        space=["x", "y", "width", "height"],
    ),
)

# %%%%%%%%%%%%%%%%%%%%
# Inspect the dataset

print(ds)

# get all annotations in image 4
ds.bbox.sel(image_id=4)


# get the bbox coordinates of the annotation with global ID = 2
ds.bbox.where(ds.global_id == 2, drop=True)

# get the global ID of the third annotation per image
ds.global_id.sel(annotation_image_id=3)

# %%
