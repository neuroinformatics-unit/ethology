from importlib.metadata import PackageNotFoundError, version

import xarray as xr

# Set xarray options
# show collapsed attributes by default
xr.set_options(display_expand_attrs=False)

try:
    __version__ = version("ethology")
except PackageNotFoundError:
    # package is not installed
    pass
