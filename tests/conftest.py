"""Pytest configuration file with shared fixtures across all tests."""

from collections.abc import Callable
from pathlib import Path

import pooch
import pytest

# load fixtures defined as modules
pytest_plugins = [
    "tests.fixtures.annotations",
]

GIN_TEST_DATA_REPO = (
    "https://gin.g-node.org/neuroinformatics/ethology-test-data"
)


@pytest.fixture(scope="session")
def pooch_registry() -> pooch.Pooch:
    """Return pooch registry with the test data.

    This fixture is common to the entire test session. This means that the
    file registry is downloaded fresh for every test session.

    Returns
    -------
    pooch.Pooch
        A Pooch object that holds the URLs and hashes for the test data files
        stored on the GIN repository.

    """
    # Cache the test data in the user's home directory
    test_data_dir = Path.home() / ".ethology-test-data"

    # Remove the file registry if it exists
    # (required in order to download it from scratch every time)
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
def get_paths_test_data() -> Callable[[dict, str], dict]:
    """Get paths of the test data files under a specific subdirectory in the
    GIN repository.

    This fixture is a factory of fixtures. It returns a function that can be
    used to create a fixture that is a dictionary holding the paths under the
    given ``subdir_name``.
    """

    def _get_paths_test_data(
        pooch_registry: pooch.Pooch, subdir_name: str
    ) -> dict:
        """Return the paths of the test files under the specified subdirectory.

        Parameters
        ----------
        pooch_registry : pooch.Pooch
            Pooch registry with the test data.
        subdir_name : str
            Name of the subdirectory under test_data for which to get the
            paths.

        Returns
        -------
        dict
            Dictionary with the requested filenames as keys and the paths as
            values.

        Notes
        -----
        The name of the subdirectories is intended to match a testing module.
        For example, to get the paths of the files used to test the annotations
        module, we call ``get_paths_test_data(pooch_registry,
        "test_annotations")``. This assumes that in the GIN repository there is
        a subdirectory named ``test_annotations`` under the ``test_data``
        directory with the relevant files.

        """
        filename_to_path = {}

        # In the pooch registry, each file is indexed by its path relative to
        # the test_data directory.
        for relative_filepath in pooch_registry.registry:
            if relative_filepath.startswith(f"{subdir_name}/"):
                fetched_filepath = Path(
                    pooch_registry.fetch(
                        relative_filepath,  # under test_data
                        progressbar=True,
                    )
                )

                filename_to_path[fetched_filepath.name] = fetched_filepath

        return filename_to_path

    return _get_paths_test_data
