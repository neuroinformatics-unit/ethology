"""JSON schemas for manual annotations files.

We use JSON schemas to validate the types of a supported
annotation file.

Note that the schema validation only checks the type of a key
if that key is present. It does not check for the presence of
the keys.

References
----------
- https://github.com/python-jsonschema/jsonschema
- https://json-schema.org/understanding-json-schema/
- https://cocodataset.org/#format-data
- https://gitlab.com/vgg/via/-/blob/master/via-2.x.y/CodeDoc.md?ref_type=heads#description-of-via-project-json-file

"""

# The VIA schema corresponds to the
# format exported by VGG Image Annotator 2.x.y
# for manual labels
VIA_SCHEMA = {
    "type": "object",
    "properties": {
        # settings for the browser-based UI of VIA
        "_via_settings": {
            "type": "object",
            "properties": {
                "ui": {"type": "object"},
                "core": {"type": "object"},
                "project": {"type": "object"},
            },
        },
        # annotations data per image
        "_via_img_metadata": {
            "type": "object",
            "additionalProperties": {
                # Each image under _via_img_metadata is indexed
                # using a unique key: FILENAME-FILESIZE.
                # We use "additionalProperties" to allow for any
                # key name, see https://stackoverflow.com/a/69811612/24834957
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "size": {"type": "integer"},
                    "regions": {
                        "type": "array",  # 'regions' is a list of dicts
                        "items": {
                            "type": "object",
                            "properties": {
                                "shape_attributes": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "x": {"type": "integer"},
                                        "y": {"type": "integer"},
                                        "width": {"type": "integer"},
                                        "height": {"type": "integer"},
                                    },
                                },
                                "region_attributes": {"type": "object"},
                            },
                        },
                    },
                    "file_attributes": {"type": "object"},
                },
            },
        },
        # _via_image_id_list contains an
        # ordered list of image keys using a unique key: FILENAME-FILESIZE,
        # the position in the list defines the image ID
        "_via_image_id_list": {
            "type": "array",
            "items": {"type": "string"},
        },
        # region attributes and file attributes, to
        # display in VIA's UI and to classify the data
        "_via_attributes": {
            "type": "object",
            "properties": {
                "region": {"type": "object"},
                "file": {"type": "object"},
            },
        },
        # version of the VIA tool used
        "_via_data_format_version": {"type": "string"},
    },
}

# The COCO schema follows the COCO dataset
# format for object detection
# See https://cocodataset.org/#format-data
COCO_SCHEMA = {
    "type": "object",
    "properties": {
        "info": {"type": "object"},
        "licenses": {
            "type": "array",
        },
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_name": {"type": "string"},
                    "id": {"type": "integer"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                },
            },
        },
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "image_id": {"type": "integer"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    # (box coordinates are measured from the
                    # top left image corner and are 0-indexed)
                    "category_id": {"type": "integer"},
                    "area": {"type": "number"},
                    # float according to the official schema
                    "iscrowd": {"type": "integer"},
                    # 0 or 1 according to the official schema
                },
            },
        },
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "supercategory": {"type": "string"},
                },
            },
        },
    },
}
