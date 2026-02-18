# Main handler for ReID in ethology

# Thin wrapper to use BoxMOT ReID models in ethology
from pathlib import Path

import numpy as np

# Import ethology's local ReID handler
from ethology.reid.core.reid_handler import ReID as EthologyReID


class ReIDHandler:
    """Ethology ReID handler using local models and backends."""

    def __init__(self, weights: str | Path, device="cpu", half=False):
        self.model = EthologyReID(weights=weights, device=device, half=half)

    def extract_features(
        self, frame: np.ndarray, dets: np.ndarray
    ) -> np.ndarray:
        """Extract feature embeddings for detections in a frame.

        Parameters
        ----------
        frame : np.ndarray
            (H, W, C) BGR image.
        dets : np.ndarray
            (N, 6) array of detections (x1, y1, x2, y2, conf, cls).

        Returns
        -------
        np.ndarray
            (N, D) feature embeddings.

        """
        return self.model(frame, dets)
