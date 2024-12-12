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

    # Remove the file registry if it exists
    # otherwise it is not downloaded from scratch every time
    file_registry_path = test_data_dir / "files-registry.txt"
    if file_registry_path.is_file():
        Path(file_registry_path).unlink()

    # Initialise pooch registry
    registry = pooch.create(
        test_data_dir,
        base_url=f"{GIN_TEST_DATA_REPO}/raw/master/test_data",
    )

    # Download only the registry file from GIN
    file_registry = pooch.retrieve(
        url=f"{GIN_TEST_DATA_REPO}/raw/master/files-registry.txt",
        known_hash=None,
        fname=file_registry_path.name,
        path=file_registry_path.parent,
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

    def _get_paths_test_data(pooch_registry, subdir_name: str) -> dict:
        """Return the paths of the test files under the specified zip filename.

        subdir_name is the name of the subdirectory under `test_data`.
        """
        test_filename_to_path = {}
        for relative_filepath in pooch_registry.registry:
            # relative to test_data
            if relative_filepath.startswith(f"{subdir_name}/"):
                # fetch file from pooch registry
                fetched_filepath = pooch_registry.fetch(
                    relative_filepath,  # under test_data
                    progressbar=True,
                )

                test_filename_to_path[Path(fetched_filepath).name] = Path(
                    fetched_filepath
                )
        return test_filename_to_path

    return _get_paths_test_data
