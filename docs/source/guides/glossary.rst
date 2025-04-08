.. glossary::
   :sorted:
This glossary defines key terms used within the ``ethology`` package and relevant concepts from computer vision and animal behaviour analysis.

Annotation
    A manual label or piece of information added to data (e.g., an image or video frame) to mark features of interest. In ```ethology```, this often refers to bounding boxes marking the location of animals. See also: *Ground Truth*, *Label*.

Any-Point Tracker
   A type of tracker, often used for *Pose Estimation*, that aims to track the location of arbitrary, user-defined points (keypoints) on an object or animal across video frames. (Planned feature).

Background Subtraction
   A computer vision technique used to separate foreground objects (e.g., moving animals) from the static background in a video sequence. (Planned feature).

Bounding Box
   A rectangular box defined by coordinates (e.g., top-left corner ``x_min``, ``y_min``, plus ``width`` and ``height``) used to indicate the location and extent of an object in an image.

Category
   The specific class or type of an annotated object (e.g., "mouse", "crab", "fish"). See also: *Supercategory*.

Category ID
    An integer identifier representing a specific *Category*, typically stored as ``category_id`` in the annotation *DataFrame*. In ``ethology``, this is a 0-based index derived automatically from the unique category names.

COCO
    Short for *Common Objects in Context*, a popular large-scale dataset and JSON-based *Annotation* format for object detection, segmentation, and captioning. ``ethology`` supports loading annotations from and saving annotations to COCO format.

DataFrame
    Refers to the internal data structure (specifically, a ``pandas.DataFrame``) used by ``ethology`` to represent annotations, particularly bounding boxes. Often referred to as the "Annotation DataFrame", it typically includes columns like ``image_filename``, ``image_id``, ``category_id``, ``x_min``, ``y_min``, ``width``, ``height``, etc., and is indexed by ``annotation_id``.

Dataset
   A collection of data, often including images or videos along with their corresponding *Annotations*. (Planned feature: ``ethology`` aims to provide tools for managing datasets).

Detector / Object Detection
   A computer vision model or algorithm designed to identify the presence and location (often via *Bounding Boxes*) of objects belonging to certain *Categories* within an image. (Planned feature).

Frame
   A single image within a video sequence.

Ground Truth
   The reference data, typically created through manual *Annotation*, that represents the correct labels or information for a given task. Used for training and evaluating models.

ID Tracker / Tracking
   A computer vision model or algorithm that follows detected objects across multiple video *Frames*, assigning a consistent identity (ID) to each object over time. (Planned feature).

Image ID
    An integer identifier representing a specific image file, stored as ``image_id`` in the annotation *DataFrame*. In ``ethology``, this is a 0-based index derived automatically from the alphabetically sorted list of unique image filenames across all loaded annotation files.

Inference
   The process of using a trained machine learning model (like a *Detector* or *Tracker*) to make predictions on new, unseen data.

JSON Schema
   A formal specification defining the expected structure, data types, and constraints for a JSON document. ``ethology`` uses JSON schemas internally to help validate input *COCO* and *VIA* annotation files.

Keypoint
   A specific point of interest on an object, often anatomical landmarks used in *Pose Estimation* (e.g., nose, ear, tail base).

Label
   Often used interchangeably with *Annotation*. Information assigned to data.

Pose Estimation
   A computer vision task aimed at identifying and localizing the *Keypoints* of an object (e.g., an animal's body parts) to determine its posture or configuration. See also: *Any-Point Tracker*. (Planned feature).

Segmentation / Instance Segmentation
   A computer vision task that involves identifying the precise pixel boundaries for each individual object instance in an image, going beyond a simple *Bounding Box*. (Planned feature).

Supercategory
   A higher-level category that groups multiple specific *Categories* (e.g., "animal" could be a supercategory for "mouse" and "crab"). Follows COCO conventions.

Training
   The process of teaching a machine learning model (e.g., a *Detector*) to perform a task by showing it examples from a labeled *Dataset* (containing *Ground Truth* annotations).

Tracker
   See *ID Tracker* or *Any-Point Tracker*. (Planned feature).

Validation
    The process of evaluating a trained machine learning model's performance on a separate portion of the *Dataset* (commonly called the "validation set") that was not used during *Training*, to estimate how well it will generalize to new data.

Validator
   In ``ethology``, refers to internal components that check if input annotation files (*VIA* or *COCO*) conform to expected rules and *JSON Schemas*.

VIA
    Short for *VGG Image Annotator*, a manual *Annotation* tool for images and videos. ``ethology`` supports loading annotations created with VIA (specifically v2.x JSON format).

Video Utilities
   Tools and functions for reading, writing, and processing video files (e.g., extracting *Frames*, getting video properties). (Planned feature).
