"""Pytest fixtures shared across annotations tests."""

from collections.abc import Callable

import pooch
import pytest


@pytest.fixture()
def annotations_test_data(
    pooch_registry: pooch.Pooch, get_paths_test_data: Callable
) -> dict:
    """Return the paths of the test files under the annotations subdirectory
    in the GIN test data repository.
    """
    return get_paths_test_data(pooch_registry, "test_annotations")
