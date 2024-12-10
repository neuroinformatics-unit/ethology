# %%

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
# read as dict
coco_data = read_json_file_as_dict(coco_file_path)

print(
    coco_data.keys()
)  # dict_keys(['annotations', 'categories', 'images', 'info', 'licenses'])
# %%%%%%%%%%%%%%%%%%%%


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

    # pad
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
# Format bboxes data as xarray DataArray

# # get bboxes coordinates per image
# bbox_and_image_id_array = np.array(
#     [annot["bbox"] + [annot["image_id"]]
# for annot in coco_data["annotations"]]
# )
# bbox_array_per_image_id = np.split(
#     bbox_and_image_id_array[:, :4],
#     np.where(np.diff(bbox_and_image_id_array[:, -1]))[0] + 1,
#     axis=0,
# )

# # pad missing annnotation-image-ids with np.nan
# max_bboxes_per_image = max([d.shape[0] for d in bbox_array_per_image_id])
# bbox_array_per_image_id_with_nans = np.stack(
#     [
#         np.concat(
#             (
#                 d,
#                 np.full(
#                     (max_bboxes_per_image - d.shape[0], d.shape[1]), np.nan
#                 ),
#             )
#         ).squeeze()
#         for d in bbox_array_per_image_id
#     ],
#     axis=1
# )  #annotation_image_id, image_id, space

# define bboxes data array
bboxes_data = compute_homog_data_array_per_image_id(
    "bbox", axis_image_id_in_output=1
)
bboxes_da = xr.DataArray.from_dict(
    {
        "dims": [
            "annotation_image_id",
            "image_id",
            "space",
        ],
        "data": bboxes_data,
        "coords": {
            "annotation_image_id": {
                "dims": "annotation_image_id",
                "data": list(range(bboxes_data.shape[0])),  # ---------
            },
            "image_id": {
                "dims": "image_id",
                "data": np.unique(
                    [annot["image_id"] for annot in coco_data["annotations"]]
                ),
            },
            "space": {
                "dims": "space",
                "data": ["x", "y", "width", "height"],
            },
        },
        # "attrs": {"title": "air temperature"},
        "name": "bbox",
    }
)

# %%%%%%%%%%%%%%%%%%%%
# Format annotation ID as xarray DataArray

# # get data
# annot_and_image_id_array = np.array(
#     [
#         [annot["id"]] + [annot["image_id"]]
#         for annot in coco_data["annotations"]
#     ],
#     dtype=int,
# )

# # split based on image id
# annot_array_per_image_id = np.split(
#     annot_and_image_id_array[:, 0].reshape(-1, 1),
#     np.where(np.diff(annot_and_image_id_array[:, -1]))[0] + 1,
#     axis=0,
# )

# # pad missing annnotation-image-ids with np.nan
# # max_bboxes_per_image = max([d.shape[0] for d in annot_array_per_image_id])
# annot_array_per_image_id_with_nans = np.stack(
#     [
#         np.concat(
#             (
#                 d,
#                 np.full(
#                     (max_bboxes_per_image - d.shape[0], d.shape[1]), np.nan
#                 ),
#             )
#         ).squeeze()
#         for d in annot_array_per_image_id
#     ],
#     axis=-1,
#     # dtype=int
# )  # annotation_image_id, image_id


# define annot ID data array
annot_ID_data = compute_homog_data_array_per_image_id(
    "id", axis_image_id_in_output=-1
)
annotation_id_da = xr.DataArray.from_dict(
    {
        "dims": [
            "annotation_image_id",
            "image_id",
        ],
        "data": annot_ID_data,
        "coords": {
            "annotation_image_id": {
                "dims": "annotation_image_id",
                "data": list(range(annot_ID_data.shape[0])),  # ---------
            },
            "image_id": {
                "dims": "image_id",
                "data": np.unique(
                    [annot["image_id"] for annot in coco_data["annotations"]]
                ),
            },
        },
        "attrs": {"title": "annotations ID per dataset"},
        "name": "bbox",
    }
)


# %%
ds = xr.Dataset(
    data_vars=dict(
        bbox=(["annotation_image_id", "image_id", "space"], bboxes_da.data),
        global_id=(
            ["annotation_image_id", "image_id"],
            annotation_id_da.data,
        ),
        # category=(["annotation_id", "category_id"], category_da),
    ),
    coords=dict(
        annotation_image_id=bboxes_da.coords["annotation_image_id"],
        image_id=bboxes_da.coords["image_id"],
        space=bboxes_da.coords["space"],
        # category_id=category_da.coords["category_id"],
    ),
    # attrs=dict(description="Weather related data."),
)

# %%%%%%%%%%%%%%%%%%%%
# Inspect the dataset

print(ds)

# get all annotations in image 4
ds.bbox.sel(image_id=4)


# get the bbox coordinates of the annotation with global ID = 2
# a.where(a.x + a.y < 4)
ds.bbox.where(ds.global_id == 2, drop=True)

# get the global ID of the third annotation per image
ds.global_id.sel(annotation_image_id=3)
