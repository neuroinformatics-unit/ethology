# Guide: Labelling Bounding Boxes with VIA

This guide provides instructions and best practices for labelling bounding box annotations using the VGG Image Annotator (VIA) tool (specifically version 2.0.12), ensuring compatibility with the `ethology` package.

While VIA offers flexibility, following these recommendations is crucial for successful data import and validation within `ethology`, especially when using the COCO export format.

## Getting VIA

1.  Download the recommended version **VIA 2.0.12** from the official website:
    *   Direct Link: [via-2.0.12.zip](https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via-2.0.12.zip)
    *   Or find it under `Downloads > Version 2` on the [VIA Software Page](https://www.robots.ox.ac.uk/~vgg/software/via/).
2.  Unzip the downloaded file.
3.  Launch VIA by opening the `via.html` file in your web browser. It runs locally and offline.

## Basic Project Setup

When starting a new project:

1.  Click the gear icon (`Settings`) in the top bar.
2.  Set a descriptive `Project Name` (e.g., `my_video_experiment_labels`).
3.  Set the `Default Path` to the directory containing your images.
    > **Important:** Ensure the path ends with a trailing slash (`/` or `\`)!
    > Example: `/path/to/my/images/`
4.  Click `Save`. (Note: You might still need to manually select images if the default path doesn't load them automatically).

## Loading Images

*   If images don't load automatically, use the `Project > Add Files` button (or the `Add Files` button in the left panel) to select the image files you want to annotate.

## Drawing Bounding Boxes

1.  **Ensure Shape:** Select `Rectangular` under `Region Shape` in the left panel.
2.  **Draw:** Click and drag on the image to draw bounding boxes around your objects of interest (e.g., animals).
3.  **Adjust:** Click a box to select it. Drag its center to move it or its borders/corners to resize it. Press `ESC` to deselect.
4.  **Copy/Paste Workflow (Optional Speedup):**
    *   Select boxes in the current frame (press `a` to select all).
    *   Copy them (press `c`).
    *   Move to the next frame (right arrow key).
    *   Paste the boxes (press `v`).
    *   Adjust positions and sizes as needed. Delete unnecessary boxes (`d`).
5.  **Key Shortcuts:**
    *   `a`: Select all boxes in the current frame.
    *   `c`: Copy selected box(es).
    *   `v`: Paste copied box(es).
    *   `d`: Delete selected box(es).
    *   `b`: Toggle visibility of boxes.
    *   `l`: Toggle visibility of box IDs/labels.
    *   `spacebar`: Toggle the annotation editor panel.
    *   Left/Right Arrows: Navigate between images.

> **Caution:** VIA v2.0.12 does not have an undo function! Save your project frequently (see Exporting section). Also, be careful not to accidentally press `v` after using the copy-paste workflow, as it will paste the boxes again.

## Defining Attributes (CRITICAL for Ethology)

To ensure your annotations can be correctly interpreted, especially when exporting to COCO format for use with `ethology`, you **must** define region attributes:

1.  **Go to Attributes:** In the left panel, click the `Attributes` tab.
2.  **Select Region Attributes:** Ensure the `Region Attributes` sub-tab is selected.
3.  **Add Supercategory:**
    *   In the `Attribute Name` text field, enter the general supercategory for your objects. We strongly recommend using `animal` for consistency with standard datasets.
    *   Click the `+` button next to the field. This adds `animal` (or your chosen name) to the list of defined attributes.
4.  **Set Type to Dropdown:**
    *   In the fields that appear for the `animal` attribute, click on `Type` and select `dropdown`.
5.  **Add Category:**
    *   In the table below the attribute settings, you need to define the specific categories belonging to the supercategory.
    *   Enter an `id` for your specific category (e.g., `0`). **Note:** While VIA allows strings here, using a simple integer like `0` is recommended. This ID is primarily for VIA's internal use; `ethology` will assign its own standardized `category_id` on import.
    *   Enter a descriptive `description` (this is the actual category name, e.g., `crab`, `mouse`).
    *   **Crucially, select the radio button under `def` (default).** This ensures that all bounding boxes you draw are automatically assigned this category (`crab` in this example). If you don't set a default, annotations might not be correctly categorized upon export.

Now, when you draw or select a bounding box, you should see a dropdown allowing you to assign the category (though it will default to the one you marked `def`).

## Exporting Annotations & Handling Gotchas

It's recommended to save your work in **two formats**:

1.  **VIA JSON:** Saves the entire project state, allowing you to reload and continue editing later.
    *   Go to `Project > Save Project`.
    *   Leave defaults `ON` and click `OK`.
    *   This saves a `<project_name>.json` file. **Save this frequently!**
2.  **COCO JSON:** The preferred format for importing annotations into `ethology`.
    *   Go to `Annotation > Export Annotations (COCO format)`.
    *   This downloads a `<project_name>_COCO.json` file (usually to your Downloads folder).

**Important Considerations & Gotchas for `ethology`:**

*   **Gotcha 1: COCO Export Requires Defined Categories**
    *   You **must** define attributes and assign categories (as described in the "Defining Attributes" section) before exporting to COCO format.
    *   **Why?** If categories are missing, the VIA export creates empty lists for `"annotations"` and `"categories"` in the COCO JSON file. This will cause a `ValueError` when validating the file in `ethology` (related to PR #39 checks).

*   **Gotcha 2: COCO Category IDs**
    *   The COCO standard requires integer `category_id`s.
    *   If you defined your category ID as a *string* in VIA (not recommended), the exported COCO file will have `null` for that `category_id`, which will likely cause errors.
    *   If you defined your category ID as an *integer* in VIA (e.g., `0`), the COCO export process itself forces it to be a **1-based index** in the exported *file*.
    *   **Note:** Don't worry too much about the exact ID value exported to COCO. When you load the COCO file into `ethology` using `from_files`, it generates its own standardized, **0-based `category_id`** based on the category names (`description` field you set in VIA).

*   **Gotcha 3: VIA Format Flexibility**
    *   The VIA JSON format (`Project > Save Project`) *does* allow saving annotations without categories or using string IDs. However, if your primary goal is to use the data with `ethology`, relying on this flexibility can lead to problems when you later try to export or use the COCO format.

*   **Gotcha 4: Image ID Differences**
    *   The `image_id` assigned in the exported files can differ:
        *   **VIA JSON:** Usually 0-indexed based on the order images were loaded/listed internally.
        *   **COCO JSON:** VIA attempts to infer an ID from the image filename (e.g., parsing numbers). If it fails, it defaults to a 1-indexed ID.
    *   **Solution:** `ethology` standardizes this on import. The `load_bboxes.from_files` function ignores the IDs in the file and assigns its own consistent, **0-based `image_id`** based on the alphabetical sorting of unique image filenames across all loaded annotation files.

## Reloading an Unfinished Project

*   Launch VIA.
*   Go to `Project > Load` and select your saved VIA JSON file (`<project_name>.json`).
*   **Troubleshooting:** If images don't load correctly, you may need to manually re-select them using `Project > Add Files` even if the `Default Path` seems correct.

## Recommendation for Ethology Users

*   **Primary Format for `ethology`:** Use the **COCO JSON** export (`Annotation > Export Annotations (COCO format)`) for importing into `ethology`, as it's a more standard format and `ethology`'s loaders/validators are primarily tested with it. Be mindful of the gotchas above during export.
*   **Backup & Editing Format:** Regularly save the **VIA JSON** (`Project > Save Project`) as your working file. Use this to reload the project in VIA if you need to continue labelling or make corrections.
