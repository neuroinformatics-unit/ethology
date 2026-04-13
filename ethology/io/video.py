"""Functions for reading and extracting frames from video files."""

from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np


def _validate_time_interval(
    start_time: float,
    end_time: float,
    duration: float,
) -> None:
    """Validate the requested time interval against the video duration.

    Parameters
    ----------
    start_time : float
        Start of the sampling interval in seconds.
    end_time : float
        End of the sampling interval in seconds.
    duration : float
        Total duration of the video in seconds.

    Raises
    ------
    ValueError
        If ``start_time`` is negative, ``end_time`` exceeds the video
        duration, or ``start_time`` is greater than or equal to
        ``end_time``.

    """
    if start_time < 0.0:
        raise ValueError(
            f"'start_time' must be non-negative, got {start_time}."
        )
    if end_time > duration:
        raise ValueError(
            f"'end_time' ({end_time} s) exceeds the video "
            f"duration ({duration:.3f} s)."
        )
    if start_time >= end_time:
        raise ValueError(
            f"'start_time' ({start_time} s) must be strictly "
            f"less than 'end_time' ({end_time} s)."
        )


def extract_frames_uniform(
    video_path: Path | str,
    n_frames: int,
    output_dir: Path | str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[Path]:
    """Extract evenly spaced frames from a video file.

    Frames are sampled at timestamps computed using
    :func:`numpy.linspace` over the interval
    ``[start_time, end_time]``. Each frame is saved as a JPEG image
    whose filename encodes both its index and its timestamp.

    Parameters
    ----------
    video_path : pathlib.Path or str
        Path to the input video file.
    n_frames : int
        Number of frames to extract. Must be at least 1.
    output_dir : pathlib.Path or str
        Directory in which to save the extracted frame images.
        It is created automatically if it does not already exist.
    start_time : float, optional
        Start of the sampling interval in seconds. If ``None``,
        defaults to ``0.0`` (the beginning of the video).
    end_time : float, optional
        End of the sampling interval in seconds. If ``None``,
        defaults to the total duration of the video.

    Returns
    -------
    list of pathlib.Path
        Paths to the saved JPEG frame images, ordered from the
        earliest to the latest extracted timestamp.

    Raises
    ------
    FileNotFoundError
        If ``video_path`` does not point to an existing file.
    ValueError
        If ``n_frames`` is less than 1.
    ValueError
        If the video file cannot be opened by OpenCV.
    ValueError
        If ``start_time`` is negative.
    ValueError
        If ``end_time`` exceeds the video duration.
    ValueError
        If ``start_time`` is greater than or equal to ``end_time``.

    Notes
    -----
    Output files are named ``frame_NNNN_tT.TTTs.jpg``, where
    ``NNNN`` is the zero-padded extraction index and ``T.TTT`` is
    the timestamp in seconds.

    Examples
    --------
    Extract 5 evenly spaced frames from the entire video:

    >>> from pathlib import Path
    >>> from ethology.io.video import extract_frames_uniform
    >>> paths = extract_frames_uniform(
    ...     video_path="recording.mp4",
    ...     n_frames=5,
    ...     output_dir="frames/",
    ... )

    Extract 10 frames from the two-second window [1.0, 3.0] s:

    >>> paths = extract_frames_uniform(
    ...     video_path="recording.mp4",
    ...     n_frames=10,
    ...     output_dir="frames/",
    ...     start_time=1.0,
    ...     end_time=3.0,
    ... )

    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: '{video_path}'.")
    if n_frames < 1:
        raise ValueError(f"'n_frames' must be at least 1, got {n_frames}.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video file: '{video_path}'. "
            "The file may be corrupted or use an unsupported codec."
        )

    try:
        fps: float = cap.get(cv2.CAP_PROP_FPS)
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration: float = total_frames / fps

        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = duration

        _validate_time_interval(start_time, end_time, duration)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamps: np.ndarray = np.linspace(start_time, end_time, n_frames)

        output_paths: list[Path] = []
        for i, t in enumerate(timestamps):
            frame_idx: int = min(int(t * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ret, frame = cap.read()
            if ret:
                filename = f"frame_{i:04d}_t{t:.3f}s.jpg"
                frame_path = output_dir / filename
                cv2.imwrite(str(frame_path), frame)
                output_paths.append(frame_path)
    finally:
        cap.release()

    return output_paths
