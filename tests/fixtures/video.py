"""Pytest fixtures shared across video tests."""

import cv2  # type: ignore[import-untyped]
import numpy as np
import pytest


@pytest.fixture()
def sample_video(tmp_path: pytest.TempPathFactory) -> dict:
    """Create a synthetic video file for testing.

    The video has 30 frames at 10 fps (3 seconds total),
    with a 64x64 resolution. Each frame has a distinct colour
    so that extraction results can be verified visually.

    Returns
    -------
    dict
        A dictionary with keys:
        ``path`` (pathlib.Path), ``fps`` (float),
        ``n_frames`` (int), ``duration`` (float),
        ``width`` (int), ``height`` (int).

    """
    fps = 10.0
    n_frames = 30
    width, height = 64, 64
    duration = n_frames / fps

    video_path = tmp_path / "sample_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(n_frames):
        # Each frame has a unique blue-channel value so frames
        # are visually distinguishable.
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * i / n_frames)  # blue
        frame[:, :, 1] = 100  # green (constant)
        frame[:, :, 2] = 50  # red   (constant)
        writer.write(frame)

    writer.release()

    return {
        "path": video_path,
        "fps": fps,
        "n_frames": n_frames,
        "duration": duration,
        "width": width,
        "height": height,
    }
