## JSON schemas for manual annotations.

[JSON files](https://www.json.org) are useful to represent a collection of key-value pairs and often used to represent manually annotated data. We use JSON schemas (via [`jsonschema`](https://github.com/python-jsonschema/jsonschema)) to validate the types of key-value pairs in supported annotation files.

For each supported JSON file format, we provide a schema that specifies the expected type for each of the expected keys in the JSON data file. The schemas are JSON files themselves, and provided under `annotations/json_schemas/`.

Note that the schema validation only checks the type of a key if that key is present in the JSON data file. It does not check for the presence of the keys.

The schema itself can also be validated for correctness, by defining a "meta-schema" under `$schema`. If no meta-schema is provided, the schema will be validated against the latest released draft of the JSON schema specification.

### VIA schema

The VIA schema refers to the JSON file format exported by the VGG Image Annotator 2.x.y (VIA) tool, used to represent bounding boxes annotations.

Each image under `_via_img_metadata` is indexed using a unique image key, made up as `<filename>-<filesize>`. We use the field `"additionalProperties"` to allow for any key name.

The section `_via_image_id_list` contains an ordered list of `<filename>-<filesize>` image keys. The position of an image in the list defines its image ID.

The section `_via_attributes` defines region attributes and file attributes, to display in the VIA tool UI and to classify the data.

The section `_via_data_format_version` contains the version of the VIA tool used to export the data.


### COCO schema
The COCO schema follows the [COCO dataset format](https://cocodataset.org/#format-data) for representing bounding box annotations.

Bounding box coordinates are measured from the top left corner of the image, and are 0-indexed.

#### References
----------
- https://json-schema.org/understanding-json-schema/
- https://gitlab.com/vgg/via/-/blob/master/via-2.x.y/CodeDoc.md?ref_type=heads#description-of-via-project-json-file
- https://python-jsonschema.readthedocs.io/en/stable/api/#jsonschema.validate
