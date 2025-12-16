from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pytest

from ethology.detectors.models import ObjectDetector


@pytest.mark.parametrize(
    "input_config, expected_config_function",
    [
        (
            {},
            "ethology.detectors.models.ObjectDetector._configure_model_pretrained",
        ),
        (
            {"checkpoint": "/path/to/checkpoint"},
            "ethology.detectors.models.ObjectDetector._configure_model_from_checkpoint",
        ),
    ],
    ids=[
        "without checkpoint",
        "with checkpoint",
    ],
)
def test_configure_model(input_config, expected_config_function):
    """Test the constructor delegates correctly to the weight loading fn."""
    with patch(expected_config_function) as mock_config_function:
        _model = ObjectDetector(input_config)

        # check expected function was called
        mock_config_function.assert_called_once()


def test_configure_model_pretrained():
    """Test that the requested model is loaded with pretrained weights."""
    pass


def test_configure_model_from_checkpoint():
    """Test that the requested model is loaded with the input checkpoint."""
    pass


# test with Lightning and torch checkpoints?
def test_get_model_state_dict():
    """Test that the model state dict retrieval from a checkpoint."""
    pass


# Do I need to test this?
def test_predict_step():
    pass


def test_run_inference():
    """Test that both .predict and ._format_predictions are called."""
    # sample_dataloader
    # sample_trainer

    # with patch ...
    pass


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
                        "boxes": np.array([[10, 20, 30, 40]], dtype=float),
                        "scores": np.array([0.9]),
                        "labels": np.array([1]),
                    },  # one detection in image 1
                    {
                        "boxes": np.array(
                            [
                                [14, 24, 34, 44],
                                [15, 25, 35, 45],
                            ],
                            dtype=float,
                        ),
                        "scores": np.array([0.95, 0.85]),
                        "labels": np.array([1, 0]),
                    },  # two detections in image 2
                    {
                        "boxes": np.array(
                            [
                                [18, 28, 38, 48],
                                [19, 29, 39, 49],
                                [20, 30, 40, 50],
                                [21, 31, 41, 51],
                            ],
                            dtype=float,
                        ),
                        "scores": np.array([0.91, 0.81, 0.71, 0.61]),
                        "labels": np.array([0, 0, 1, 1]),
                    },  # four detections in image 3
                ],
                # Batch 1: 2 images
                [
                    {
                        "boxes": np.array(
                            [[22, 32, 42, 52], [23, 33, 43, 53]],
                            dtype=float,
                        ),
                        "scores": np.array([0.92, 0.82]),
                        "labels": np.array([1, 1]),
                    },  # 2 detections in image 1
                    {
                        "boxes": np.array(
                            [[26, 36, 46, 56], [27, 37, 47, 57]],
                            dtype=float,
                        ),
                        "scores": np.array([0.93, 0.83]),
                        "labels": np.array([0, 1]),
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
                        "boxes": np.array([[10, 20, 30, 40]], dtype=float),
                        "scores": np.array([0.95]),
                        "labels": np.array([2]),
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
                        "boxes": np.array([[10, 20, 30, 40]], dtype=float),
                        "scores": np.array([0.95]),
                        "labels": np.array([2]),
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
