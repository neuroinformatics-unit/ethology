"""Example notebook showing how to use `movement` with annotations."""

# %%
import matplotlib.pyplot as plt
from movement.plots import plot_occupancy
from movement.roi import PolygonOfInterest

from ethology.annotations.io import load_bboxes

# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
df = load_bboxes.from_files(
    (
        "/Users/sofia/.ethology-test-data/test_annotations/"
        "medium_bboxes_dataset_VIA/VIA_JSON_sample_2.json"
    ),
    format="VIA",
)
ds = load_bboxes._df_to_xarray_ds(df)

ds_as_movement = ds.rename_dims({"image_id": "time", "id": "individuals"})

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot labels occupancy
fig, ax, hist = plot_occupancy(ds_as_movement.position, bins=[100, 50])
ax.set_label("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()

# # add colorbar
# cbar = plt.colorbar()
# cbar.set_label('Occupancy')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot labels

# plot_centroid_trajectory(ds_as_movement.position) ---odd

fig, ax = plt.subplots()
ax.scatter(
    ds.position.sel(space="x").values,
    ds.position.sel(space="y").values,
    s=1,
    alpha=0.5,
)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot bboxes as ROIs?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# labels within a polygon
central_region = PolygonOfInterest(
    ((1000, 500), (1000, 1250), (2500, 1250), (2500, 500)),
    #  ((2120, 0),(2120, 80), (2220, 80), (2220, 0)),
    # there should be 4 inside here ^
    name="Central region",
)

# plot on occupancy map
central_region.plot(ax, facecolor="red", alpha=0.25)

# OJO ds.position contains nans -- they are not in the polygon
print(ds.position.isnull().sum())

# check labels in the polygon
ds_in_region = central_region.contains_point(ds.position)
print(ds_in_region.sum())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# For kpts: plot in egocentric coord syst
