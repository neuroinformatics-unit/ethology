from typing import Any

import torch
import xarray as xr


def run_tracker_on_detections_ds(
    detections_ds: xr.Dataset,
    tracker: Any,
    device: torch.device,
) -> xr.Dataset:
    """Run tracker on detections dataset."""
    pass