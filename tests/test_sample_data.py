import os
import pytest

from ethology.datasets import load_sample_tracking


def test_load_sample_tracking_returns_path():
    """Ensure loader returns a valid file path."""
    path = load_sample_tracking()

    assert isinstance(path, str)
    assert os.path.exists(path)


def test_load_sample_tracking_cached():
    """Ensure dataset is cached and not downloaded twice."""
    path1 = load_sample_tracking()
    path2 = load_sample_tracking()

    assert path1 == path2


def test_sample_dataset_filename():
    """Check returned dataset filename."""
    path = load_sample_tracking()

    assert path.endswith("example_tracking.csv")