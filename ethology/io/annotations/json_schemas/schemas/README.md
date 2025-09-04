## JSON schemas for manual annotations files.

We use JSON schemas to validate the types of a supported annotation file.

Note that the schema validation only checks the type of a key if that key is present. It does not check for the presence of the keys.

If the meta-schema (under $schema) is not provided, the jsonschema validator uses the the latest released draft of the JSON schema specification.

## VIA schema

The VIA schema corresponds to the format exported by VGG Image Annotator 2.x.y (VIA) for object detection annotations.

Each image under `_via_img_metadata` is indexed using a unique key: FILENAME-FILESIZE. We use "additionalProperties" to allow for any key name, see https://stackoverflow.com/a/69811612/24834957.

The section `_via_image_id_list` contains an ordered list of image keys using a unique key: `FILENAME-FILESIZE`, the position in the list defines the image ID.

The section `_via_attributes` contains region attributes and file attributes, to display in VIA's UI and to classify the data.

The section `_via_data_format_version` contains the version of the VIA tool used.


## COCO schema
The COCO schema follows the COCO dataset format for object detection, see https://cocodataset.org/#format-data.

Box coordinates are measured from the top left corner of the image, and are 0-indexed.
### References
----------
- https://github.com/python-jsonschema/jsonschema
- https://json-schema.org/understanding-json-schema/
- https://cocodataset.org/#format-data
- https://gitlab.com/vgg/via/-/blob/master/via-2.x.y/CodeDoc.md?ref_type=heads#description-of-via-project-json-file
- https://python-jsonschema.readthedocs.io/en/stable/api/#jsonschema.validate
