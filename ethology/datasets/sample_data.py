"""Utilities for downloading and loading example datasets for ethology.

This module provides helper functions that use `pooch` to fetch
small example datasets used in tutorials, tests, and documentation.
"""

import pooch

...


DATA_REGISTRY = pooch.create(
    path=pooch.os_cache("ethology"),
    base_url="https://example-dataset-url/",
    registry={"example_tracking.csv": None},
)


def load_sample_tracking() -> str:
    """Fetch a sample tracking dataset.

    Returns
    -------
    str
        Path to the downloaded dataset.

    """
    return DATA_REGISTRY.fetch("example_tracking.csv")
