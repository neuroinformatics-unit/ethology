"""JSON schemas for VIA and COCO annotations."""

VIA_UNTRACKED_SCHEMA = {
    "type": "object",
    "properties": {
        # settings for browser UI
        "_via_settings": {
            "type": "object",
            "properties": {
                "ui": {"type": "object"},
                "core": {"type": "object"},
                "project": {"type": "object"},
            },
        },
        # annotation data
        "_via_img_metadata": {
            "type": "object",
            "additionalProperties": {
                # "additionalProperties" to allow any key,
                # see https://stackoverflow.com/a/69811612/24834957
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "size": {"type": "integer"},
                    "regions": {
                        "type": "array",  # a list of dicts
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
                                    "region_attributes": {
                                        "type": "object"
                                    },  # we just check it's a dict
                                },
                            },
                        },
                    },
                    "file_attributes": {"type": "object"},
                },
            },
        },
        # ordered list of image keys
        # - the position defines the image ID
        "_via_image_id_list": {
            "type": "array",
            "items": {"type": "string"},
        },
        # region (aka annotation) and file attributes for VIA UI
        "_via_attributes": {
            "type": "object",
            "properties": {
                "region": {"type": "object"},
                "file": {"type": "object"},
            },
        },
        # version of the VIA data format
        "_via_data_format_version": {"type": "string"},
    },
}

COCO_UNTRACKED_SCHEMA = {
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
                    "id": {"type": "integer"},  # annotation global ID
                    "image_id": {"type": "integer"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "category_id": {"type": "integer"},
                    "area": {"type": "integer"},
                    "iscrowd": {"type": "integer"},
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
