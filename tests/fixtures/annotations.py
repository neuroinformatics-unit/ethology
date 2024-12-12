"""Pytest fixtures shared across annotation tests."""

import pytest


@pytest.fixture()
def annotations_test_data(pooch_registry, get_paths_test_data):
    return get_paths_test_data(pooch_registry, "test_annotations")
