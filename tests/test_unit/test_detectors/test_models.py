from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr
from torchvision.models import get_model
from torchvision.models.detection import faster_rcnn, fcos, retinanet

from ethology.detectors.models import (
    DEFAULT_NUM_CLASSES,
    ObjectDetector,
    _get_n_classes_anchor_based,
    _get_n_classes_fasterrcnn,
    _get_n_classes_in_detector,
)

# Map model names to class types
MODEL_CLASS_REGISTRY = {
    "fasterrcnn_resnet50_fpn_v2": faster_rcnn.FasterRCNN,
    "fasterrcnn_mobilenet_v3_large_fpn": faster_rcnn.FasterRCNN,
    "fcos_resnet50_fpn": fcos.FCOS,
    "retinanet_resnet50_fpn_v2": retinanet.RetinaNet,
}


@pytest.fixture
def sample_checkpoint_path_and_classes(tmp_path: Path) -> Callable:
    def _checkpoint_path_and_classes(
        model_class, ckpt_format: str, num_classes=91
    ) -> tuple[Path, int]:
        """Return the path to a sample checkpoint.

        The checkpoint is for the requested architecture (model_class) and
        format (Pytorch or Pytorch Lightning convention). By default,
        """
        # Create a model with randomly initialised weights and save its state
        # If not specified it has 91 categories by default
        model = get_model(model_class, num_classes=num_classes)
        ckpt_filename = "test_checkpoint"

        if ckpt_format == "lightning":
            checkpoint_path = tmp_path / f"{ckpt_filename}.ckpt"

            # Save as Lightning-style checkpoint (with "model." prefix)
            state_dict = {
                f"model.{k}": v for k, v in model.state_dict().items()
            }
            torch.save({"state_dict": state_dict}, checkpoint_path)

        elif ckpt_format == "torch":
            checkpoint_path = tmp_path / f"{ckpt_filename}.pt"

            # Save the state_dict as recommended in pytorch docs
            # (see https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
            torch.save(model.state_dict(), checkpoint_path)

        else:
            raise ValueError(f"Unsupported format: {ckpt_format}")

        return checkpoint_path, num_classes

    return _checkpoint_path_and_classes


@pytest.fixture
def valid_predictions_dataset():
    """Create a valid bbox detections dataset to simulate predictions."""
    image_ids = [
        1,
        2,
    ]
    annotation_ids = [0, 1]
    space_dims = ["x", "y"]

    # Create position, shape and confidence data all zeros
    position_data = np.zeros(
        (len(image_ids), len(space_dims), len(annotation_ids))
    )
    shape_data = np.copy(position_data)
    category_data = np.ones((len(image_ids), len(annotation_ids)))
    confidence_data = np.zeros((len(image_ids), len(annotation_ids)))

    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            "position": (["image_id", "space", "id"], position_data),
            "shape": (["image_id", "space", "id"], shape_data),
            "category": (["image_id", "id"], category_data),
            "confidence": (["image_id", "id"], confidence_data),
        },
        coords={
            "image_id": image_ids,
            "space": ["x", "y"],
            "id": annotation_ids,
        },
    )

    return ds


# -------------- Gral model configuration ---------------


@pytest.mark.parametrize(
    "config, expected_exception",
    [
        (
            "",
            pytest.raises(TypeError, match="config must be a dictionary"),
        ),
        (
            {"checkpoint": "path/to/checkpoint"},
            pytest.raises(
                ValueError,
                match=("model_class must be defined in config"),
            ),
        ),
        (
            {"model_class": "foo"},
            pytest.raises(ValueError, match="Model 'foo' not supported"),
        ),
        (
            {"model_class": "fcos_resnet50_fpn", "model_kwargs": "foo"},
            pytest.raises(TypeError, match="model_kwargs must be a dict"),
        ),
        (
            {
                "model_class": "fcos_resnet50_fpn",
                "model_kwargs": {"n_classes": 3},
            },
            pytest.raises(
                ValueError,
                match=(
                    "Invalid key 'n_classes' in model_kwargs. "
                    "Did you mean 'num_classes'?"
                ),
            ),
        ),
        (
            {
                "model_class": "fcos_resnet50_fpn",
                "model_kwargs": {"num_classes": 3},
            },
            does_not_raise(),
        ),
        (
            {
                "model_class": "fcos_resnet50_fpn",
                "model_kwargs": {"trainable_backbone_layers": 5},
            },
            does_not_raise(),
        ),
        (
            {
                "model_class": "fcos_resnet50_fpn",
                "model_kwargs": {
                    "num_classes": 3,
                    "trainable_backbone_layers": 5,
                },
            },
            does_not_raise(),
        ),
    ],
    ids=[
        "config wrong type",
        "model_class missing",
        "model_class unsupported",
        "model_kwargs wrong type",
        "num_classes misspelt",
        "model_kwargs single correct 1",
        "model_kwargs single correct 2",
        "model_kwargs multiple correct",
    ],
)
def test_validate_config(config, expected_exception):
    """Test the config validation throws the expected errors."""
    with expected_exception:
        ObjectDetector._validate_config(config)


@pytest.mark.parametrize(
    "input_config, expected_config_function",
    [
        (
            {"model_class": "fcos_resnet50_fpn"},
            "ethology.detectors.models.ObjectDetector._configure_model_pretrained",
        ),
        (
            {
                "model_class": "fcos_resnet50_fpn",
                "checkpoint": "/path/to/checkpoint",
            },
            "ethology.detectors.models.ObjectDetector._configure_model_from_checkpoint",
        ),
    ],
    ids=[
        "config without checkpoint",
        "config with checkpoint",
    ],
)
def test_configure_model(input_config, expected_config_function):
    """Test the constructor delegates correctly to the weight loading fn."""
    with patch(expected_config_function) as mock_config_function:
        model = ObjectDetector(input_config)

        # check expected function was called
        mock_config_function.assert_called_once()

        # check model params are saved as attributes
        assert model._model_class  # check it is truthy
        assert model.model_params  # should have at least num_classes
        assert "num_classes" in model.model_params


@pytest.mark.parametrize(
    "model_class",
    [
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "retinanet_resnet50_fpn_v2",
        "fcos_resnet50_fpn",
    ],
)
@pytest.mark.parametrize(
    "model_kwargs, expected_num_classes",
    [
        ({"num_classes": 1}, 1),
        ({"num_classes": 100}, 100),
        ({}, DEFAULT_NUM_CLASSES),
    ],
)
def test_configure_model_pretrained_n_classes(
    model_class, model_kwargs, expected_num_classes
):
    """Test that the requested number of classes is passed to the model."""
    # Instantiate detector
    config = {
        "model_class": model_class,
        "model_kwargs": model_kwargs,
    }
    detector = ObjectDetector(config)

    # Check n of classes in output layer
    assert (
        _get_n_classes_in_detector(detector.model, model_class)
        == expected_num_classes
    )

    # Check model architecture and type
    assert isinstance(detector.model, MODEL_CLASS_REGISTRY[model_class])
    assert isinstance(detector.model, torch.nn.Module)


# -------------- Configure model from ckpt -----------


@pytest.mark.parametrize(
    "ckpt_format",
    [
        "lightning",
        "torch",
    ],
)
@pytest.mark.parametrize(
    "model_class",
    [
        "fasterrcnn_resnet50_fpn_v2",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fcos_resnet50_fpn",
        "retinanet_resnet50_fpn_v2",
    ],
)
@pytest.mark.parametrize(
    "num_classes",
    [
        None,  # use default (91)
        5,
    ],
)
def test_configure_model_from_checkpoint(
    sample_checkpoint_path_and_classes,
    model_class,
    ckpt_format,
    num_classes,
):
    """Test loading weights from a checkpoint.

    We test the cases of different checkpoint saving formats,
    different model classes, and whether the number of classes in
    the model architecture is different from the default (91 classes
    for COCO2017).
    """
    # Get ckpt for specified architecture, num_classes and ckpt format
    # If num_classes is None: model will the default number of classes
    ckpt_kwargs = {} if num_classes is None else {"num_classes": num_classes}
    ckpt_path, out_num_classes = sample_checkpoint_path_and_classes(
        model_class, ckpt_format, **ckpt_kwargs
    )

    # Define config
    input_config = {
        "model_class": model_class,
        "checkpoint": str(ckpt_path),
    }
    if num_classes is not None:
        input_config["model_kwargs"] = {"num_classes": out_num_classes}

    # Instantiate detector
    detector = ObjectDetector(input_config)

    # Check type and type
    assert isinstance(detector.model, torch.nn.Module)
    assert (
        _get_n_classes_in_detector(detector.model, model_class)
        == out_num_classes
    )


@pytest.mark.parametrize(
    "format",
    [
        "lightning",
        "torch",
    ],
)
@pytest.mark.parametrize(
    "input_config, expected_exception",
    [
        (
            {
                "model_class": "fasterrcnn_resnet50_fpn_v2",
                "model_kwargs": {"num_classes": 2},
            },
            pytest.raises(
                RuntimeError,
                match=(
                    r"Error\(s\) in loading state_dict for FasterRCNN:\n\t"
                    r"size mismatch.*"
                ),
            ),
        ),  # ckpt has 91 classes but config specifies 2
        (
            {
                "model_class": "fcos_resnet50_fpn",
            },
            pytest.raises(
                RuntimeError,
                match=(
                    r"Error\(s\) in loading state_dict for FCOS:\n\t"
                    r"Missing key\(s\) in state_dict.*"
                ),
            ),
        ),  # ckpt is fasterrcnn_resnet50_fpn_v2 but config fcos_resnet50_fpn
    ],
    ids=["mismatch_n_classes", "mismatch_architecture"],
)
def test_configure_model_from_checkpoint_invalid(
    sample_checkpoint_path_and_classes,
    format,
    input_config,
    expected_exception,
):
    """Test loading weights from a FasterRCNN checkpoint with 91 classes."""
    # Get fasterrcnn COCO2017 ckpt and add to config
    fasterrcnn_ckpt_path, _ = sample_checkpoint_path_and_classes(
        "fasterrcnn_resnet50_fpn_v2", format
    )
    input_config["checkpoint"] = str(fasterrcnn_ckpt_path)

    # Check detector instantiation throws error
    with expected_exception:
        _detector = ObjectDetector(input_config)


# -------- Convenience functions ---------------------------


@pytest.mark.parametrize(
    "model_class, expected_get_n_classes_fn",
    [
        ("fasterrcnn_resnet50_fpn_v2", "_get_n_classes_fasterrcnn"),
        ("fasterrcnn_mobilenet_v3_large_fpn", "_get_n_classes_fasterrcnn"),
        ("fcos_resnet50_fpn", "_get_n_classes_anchor_based"),
        ("retinanet_resnet50_fpn_v2", "_get_n_classes_anchor_based"),
    ],
)
def test_get_n_classes_in_detector(model_class, expected_get_n_classes_fn):
    """Test _get_n_classes_in_detector delegates to the right function."""
    model = torch.nn.Module()
    function_to_patch = (
        f"ethology.detectors.models.{expected_get_n_classes_fn}"
    )
    with patch(function_to_patch) as mock_get_n_classes_fn:
        _ = _get_n_classes_in_detector(model, model_class)

        # check expected function was called
        mock_get_n_classes_fn.assert_called_once()


def test_get_n_classes_in_detector_invalid():
    model = torch.nn.Module()
    with pytest.raises(ValueError, match="Unsupported model class: foo"):
        _ = _get_n_classes_in_detector(model, "foo")


@pytest.mark.parametrize(
    "model_class, counting_function",
    [
        ("fasterrcnn_resnet50_fpn_v2", _get_n_classes_fasterrcnn),
        ("fasterrcnn_mobilenet_v3_large_fpn", _get_n_classes_fasterrcnn),
        ("fcos_resnet50_fpn", _get_n_classes_anchor_based),
        ("retinanet_resnet50_fpn_v2", _get_n_classes_anchor_based),
    ],
)
@pytest.mark.parametrize(
    "num_classes",
    [
        2,  # minimum meaningful value (background + 1 object class)
        100,
    ],
)
def test_get_n_classes_specific_function(
    model_class, counting_function, num_classes
):
    """Test the _get_n_classes... architecture-specific functions."""
    # Instantiate model using torch convenience fn
    model = get_model(model_class, num_classes=num_classes)
    assert counting_function(model) == num_classes


@pytest.mark.parametrize(
    "checkpoint_dict",
    [
        {"layer1.weight": 1, "layer1.bias": 2},
        # state dict at first level (torch no nesting)
        {"state_dict": {"layer1.weight": 1, "layer1.bias": 2}},
        # state dict under "state_dict" key (torch nested)
        {"state_dict": {"model.layer1.weight": 1, "model.layer1.bias": 2}},
        # state dict "state_dict" key using "model." prefix
        # (nested with model. prefix, Lightning convention)
    ],
)
def test_get_model_state_dict(checkpoint_dict):
    """Test state_dict extraction from different checkpoint dict formats."""
    expected_state_dict = {"layer1.weight": 1, "layer1.bias": 2}
    state_dict_out = ObjectDetector._get_model_state_dict(checkpoint_dict)
    assert state_dict_out == expected_state_dict


# ---------- Inference ---------------------


def test_predict_step():
    """Check predict_step returns expected structure."""
    # Create a minimal detector
    config = {"model_class": "fasterrcnn_resnet50_fpn_v2"}
    detector = ObjectDetector(config)
    detector.eval()

    # Create a fake batch of 2 RGB images and empty annotations
    batch_size = 2
    images = torch.rand(batch_size, 3, 224, 224)
    annotations = {}  # unused by predict_step
    batch = (images, annotations)

    # Run predict_step
    predictions = detector.predict_step(batch, batch_idx=0)

    # Check output is a list of dicts
    assert isinstance(predictions, list)
    assert len(predictions) == batch_size
    assert all(isinstance(pred, dict) for pred in predictions)

    # Check relevant keys exist
    assert all("boxes" in pred for pred in predictions)
    assert all("scores" in pred for pred in predictions)
    assert all("labels" in pred for pred in predictions)

    # Check arrays are torch tensors
    assert all(isinstance(pred["boxes"], torch.Tensor) for pred in predictions)
    assert all(
        isinstance(pred["scores"], torch.Tensor) for pred in predictions
    )
    assert all(
        isinstance(pred["labels"], torch.Tensor) for pred in predictions
    )


def test_run_inference(valid_predictions_dataset):
    """Test that both .predict and ._format_predictions are called."""
    # Create a minimal detector
    config = {"model_class": "fasterrcnn_resnet50_fpn_v2"}
    detector = ObjectDetector(config)

    # Mock inputs to run_inference
    mock_trainer = MagicMock()
    mock_dataloader = MagicMock()
    ds_attrs = {"source": "test"}

    # Mock the output of trainer.predict
    mock_predictions = [
        [  # batch 0
            {  # image 0
                "boxes": torch.tensor(
                    [
                        [10.0, 20.0, 50.0, 60.0],
                        [15.0, 25.0, 55.0, 65.0],
                    ]
                ),
                "scores": torch.tensor([0.95, 0.87]),
                "labels": torch.tensor([1, 2]),
            }
        ]
    ]
    mock_trainer.predict.return_value = mock_predictions

    # Mock the output of detector._format_predictions
    with patch.object(
        ObjectDetector,
        "_format_predictions",
        return_value=valid_predictions_dataset,
    ) as mock_format_fn:
        result = detector.run_inference(
            mock_trainer,
            mock_dataloader,
            attrs=ds_attrs,
        )

    # Verify trainer.predict was called once
    mock_trainer.predict.assert_called_once_with(detector, mock_dataloader)

    # Verify _format_predictions was called once
    mock_format_fn.assert_called_once_with(mock_predictions, attrs=ds_attrs)

    # Check the result from run_inference is the same as _format_predictions
    assert result == valid_predictions_dataset


@pytest.mark.parametrize(
    (
        "list_raw_predictions, "
        "expected_n_images, expected_max_detections, expected_exception"
    ),
    [
        # ---------------- Two valid batches --------------------
        (
            [
                # Batch 0: 3 images
                [
                    {
                        "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                        "scores": torch.tensor([0.9]),
                        "labels": torch.tensor([1]),
                    },  # one detection in image 1.
                    # Note: float literals are torch.float32 by default
                    {
                        "boxes": torch.tensor(
                            [
                                [14.0, 24.0, 34.0, 44.0],
                                [15.0, 25.0, 35.0, 45.0],
                            ],
                        ),
                        "scores": torch.tensor([0.95, 0.85]),
                        "labels": torch.tensor([1, 0]),
                    },  # two detections in image 2
                    {
                        "boxes": torch.tensor(
                            [
                                [18.0, 28.0, 38.0, 48.0],
                                [19.0, 29.0, 39.0, 49.0],
                                [20.0, 30.0, 40.0, 50.0],
                                [21.0, 31.0, 41.0, 51.0],
                            ],
                        ),
                        "scores": torch.tensor([0.91, 0.81, 0.71, 0.61]),
                        "labels": torch.tensor([0, 0, 1, 1]),
                    },  # four detections in image 3
                ],
                # Batch 1: 2 images
                [
                    {
                        "boxes": torch.tensor(
                            [
                                [22.0, 32.0, 42.0, 52.0],
                                [23.0, 33.0, 43.0, 53.0],
                            ],
                        ),
                        "scores": torch.tensor([0.92, 0.82]),
                        "labels": torch.tensor([1, 1]),
                    },  # 2 detections in image 1
                    {
                        "boxes": torch.tensor(
                            [
                                [26.0, 36.0, 46.0, 56.0],
                                [27.0, 37.0, 47.0, 57.0],
                            ],
                        ),
                        "scores": torch.tensor([0.93, 0.83]),
                        "labels": torch.tensor([0, 1]),
                    },  # 2 detections in image 2
                ],
            ],
            5,  # expected_n_images
            4,  # expected_n_max_detections
            does_not_raise(),
        ),
        #  ---------- Single valid batch with one image and one detection -----
        (
            [
                [
                    {
                        "boxes": torch.tensor(
                            [[10.0, 20.0, 30.0, 40.0]], dtype=float
                        ),
                        "scores": torch.tensor([0.95]),
                        "labels": torch.tensor([2]),
                    }
                ],
            ],
            1,  # expected_n_images
            1,  # expected_max_detections
            does_not_raise(),
            #  ------ One valid batch with one image and one empty batch -----
        ),
        (
            [
                [
                    {
                        "boxes": torch.tensor(
                            [[10.0, 20.0, 30.0, 40.0]], dtype=float
                        ),
                        "scores": torch.tensor([0.95]),
                        "labels": torch.tensor([2]),
                    }
                ],
                [],
            ],
            1,  # expected_n_images
            1,  # expected_max_detections
            does_not_raise(),
        ),
        # ---------------- Invalid input lists ----------------
        (
            {},  # wrong type
            None,
            None,
            pytest.raises(TypeError, match="predictions must be a list"),
        ),
        (
            [],  # empty outer lists (no batches)
            None,
            None,
            pytest.raises(ValueError, match="predictions list is empty"),
        ),
        (
            [[], []],  # two empty batches, 0 images in each batch
            None,
            None,
            pytest.raises(
                ValueError, match="predictions list contains no image data"
            ),
        ),
    ],
    ids=[
        "two_batches_padding",
        "single_batch_no_padding",
        "wrong_type",
        "no_batches",
        "two_batches_both_no_images",
        "two_batches_one_no_images",
    ],
)
def test_format_predictions(
    list_raw_predictions,
    expected_n_images,
    expected_max_detections,
    expected_exception,
):
    """Test that predictions are formatted as a detections dataset."""
    with expected_exception as excinfo:
        ds = ObjectDetector._format_predictions(list_raw_predictions)

        # If valid, check array shapes match input data
        if not excinfo:
            assert ds.position.shape == (
                expected_n_images,
                2,  # space dimension
                expected_max_detections,
            )
            assert ds.shape.shape == (
                expected_n_images,
                2,  # space dimension
                expected_max_detections,
            )
            assert ds.confidence.shape == (
                expected_n_images,
                expected_max_detections,
            )
            assert ds.category.shape == (
                expected_n_images,
                expected_max_detections,
            )
