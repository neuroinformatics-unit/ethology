import pooch

DATA_REGISTRY = pooch.create(
    path=pooch.os_cache("ethology"),
    base_url="https://example-dataset-url/",
    registry={
        "example_tracking.csv": None,
    },
)


def load_sample_tracking():
    """Fetch a sample tracking dataset.

    Returns
    -------
    str
        Path to downloaded dataset.

    """
    return DATA_REGISTRY.fetch("example_tracking.csv")
