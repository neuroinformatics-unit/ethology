"""Pytest configuration file with shared fixtures across all tests."""

from pathlib import Path

import pooch
import pytest

GIN_TEST_DATA_REPO = (
    "https://gin.g-node.org/neuroinformatics/ethology-test-data"
)

pytest_plugins = [
    "tests.fixtures.annotations",
]


@pytest.fixture(scope="session")
def pooch_registry() -> dict:
    """Pooch registry for the test data.

    This fixture is common to the entire test session. The
    file registry is downloaded fresh for every test session.

    Returns
    -------
    dict
        URL and hash of the GIN repository with the test data

    """
    # Cache the test data in the user's home directory
    test_data_dir = Path.home() / ".ethology-test-data"

    # Initialise pooch registry
    registry = pooch.create(
        test_data_dir,
        base_url=f"{GIN_TEST_DATA_REPO}/raw/master/test_data",
    )

    # Download only the registry file from GIN
    # if known_hash = None, the file is always downloaded.
    file_registry = pooch.retrieve(
        url=f"{GIN_TEST_DATA_REPO}/raw/master/files-registry.txt",
        known_hash=None,
        fname="files-registry.txt",
        path=test_data_dir,
    )

    # Load registry file onto pooch registry
    registry.load_registry(file_registry)

    return registry


@pytest.fixture()
def get_paths_test_data():
    """Define a factory fixture to get the paths of the data files
    under a specific zip.

    The name of the zip file is intended to match a testing module. For
    example, to get the paths to the test files for the annotations
    tests module, we would call `get_paths_test_data(pooch_registry,
    "test_annotations")` in a test. This assumes in the GIN repository
    there is a zip file named `test_annotations.zip` under the `test_data`
    directory containing the relevant test files.
    """

    def _get_paths_test_data(pooch_registry, zip_filename: str) -> dict:
        """Return the paths of the test files under the specified zip filename.

        The zip filename is expected to match a testing module.
        """
        # Fetch the test data for the annotations module
        list_files_in_local_storage = pooch_registry.fetch(
            f"{zip_filename}.zip",
            processor=pooch.Unzip(extract_dir=""),
            progressbar=True,
        )

        # Filter out files not under `test_annotations` directory
        list_files_annotations = [
            f
            for f in list_files_in_local_storage
            if (zip_filename in f) and (not f.endswith(".zip"))
        ]

        # return paths as dict
        input_data_dict = {}
        for f in list_files_annotations:
            input_data_dict[Path(f).name] = Path(f)

        return input_data_dict

    return _get_paths_test_data
