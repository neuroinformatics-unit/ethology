from importlib.metadata import PackageNotFoundError, version
import xarray as xr
from pathlib import Path

# Set xarray attributes collapsed by default
xr.set_options(display_expand_attrs=False)

# Set cache directory for ethology package
ETHOLOGY_CACHE_DIR = Path.home() / ".ethology"
ETHOLOGY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

try:
    __version__ = version("ethology")
except PackageNotFoundError:
    # package is not installed
    pass
