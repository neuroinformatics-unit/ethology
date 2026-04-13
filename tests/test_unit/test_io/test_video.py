"""Tests for ethology.io.video."""

from pathlib import Path

import pytest

from ethology.io.video import extract_frames_uniform

# ------------------------------------------------------------------ #
#  Valid-input tests                                                   #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "n_frames",
    [1, 5, 10, 30],
    ids=["1_frame", "5_frames", "10_frames", "30_frames"],
)
def test_extract_frames_uniform_returns_correct_count(
    n_frames: int,
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """Extracting n_frames from a video returns exactly n_frames paths."""
    output_dir = tmp_path / "frames"
    paths = extract_frames_uniform(
        video_path=sample_video["path"],
        n_frames=n_frames,
        output_dir=output_dir,
    )
    assert len(paths) == n_frames


@pytest.mark.parametrize(
    "n_frames",
    [1, 5, 10],
    ids=["1_frame", "5_frames", "10_frames"],
)
def test_extract_frames_uniform_files_exist(
    n_frames: int,
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """Every path returned by extract_frames_uniform points to a real file."""
    output_dir = tmp_path / "frames"
    paths = extract_frames_uniform(
        video_path=sample_video["path"],
        n_frames=n_frames,
        output_dir=output_dir,
    )
    for p in paths:
        assert p.is_file(), f"Expected file to exist: {p}"


@pytest.mark.parametrize(
    "n_frames",
    [1, 5],
    ids=["1_frame", "5_frames"],
)
def test_extract_frames_uniform_output_filenames(
    n_frames: int,
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """Output filenames follow the pattern frame_NNNN_tT.TTTs.jpg."""
    output_dir = tmp_path / "frames"
    paths = extract_frames_uniform(
        video_path=sample_video["path"],
        n_frames=n_frames,
        output_dir=output_dir,
    )
    for i, p in enumerate(paths):
        assert p.suffix == ".jpg"
        assert p.name.startswith(f"frame_{i:04d}_t")


def test_extract_frames_uniform_output_dir_created(
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """extract_frames_uniform creates output_dir if it does not exist."""
    output_dir = tmp_path / "new_dir" / "nested_dir"
    assert not output_dir.exists()
    extract_frames_uniform(
        video_path=sample_video["path"],
        n_frames=3,
        output_dir=output_dir,
    )
    assert output_dir.is_dir()


def test_extract_frames_uniform_output_in_correct_dir(
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """All returned paths are located inside output_dir."""
    output_dir = tmp_path / "frames"
    paths = extract_frames_uniform(
        video_path=sample_video["path"],
        n_frames=5,
        output_dir=output_dir,
    )
    for p in paths:
        assert p.parent == output_dir


# ------------------------------------------------------------------ #
#  Time-interval tests                                                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "start_time, end_time, n_frames",
    [
        (0.0, 1.0, 5),
        (1.0, 2.0, 3),
        (0.5, 2.5, 8),
    ],
    ids=[
        "first_second",
        "middle_second",
        "middle_two_seconds",
    ],
)
def test_extract_frames_uniform_with_time_interval(
    start_time: float,
    end_time: float,
    n_frames: int,
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """Extracting with explicit start/end returns the correct frame count."""
    output_dir = tmp_path / "frames"
    paths = extract_frames_uniform(
        video_path=sample_video["path"],
        n_frames=n_frames,
        output_dir=output_dir,
        start_time=start_time,
        end_time=end_time,
    )
    assert len(paths) == n_frames
    for p in paths:
        assert p.is_file()


# ------------------------------------------------------------------ #
#  Invalid-input tests                                                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    "kwargs_override, expected_exception, error_fragment",
    [
        # video_path does not exist
        (
            {"video_path": "non_existent_video.mp4"},
            pytest.raises(FileNotFoundError),
            "Video file not found",
        ),
        # n_frames = 0
        (
            {"n_frames": 0},
            pytest.raises(ValueError),
            "'n_frames' must be at least 1",
        ),
        # n_frames negative
        (
            {"n_frames": -3},
            pytest.raises(ValueError),
            "'n_frames' must be at least 1",
        ),
        # start_time negative
        (
            {"start_time": -1.0},
            pytest.raises(ValueError),
            "'start_time' must be non-negative",
        ),
        # end_time beyond video duration
        (
            {"end_time": 9999.0},
            pytest.raises(ValueError),
            "'end_time'",
        ),
        # start_time equal to end_time
        (
            {"start_time": 1.0, "end_time": 1.0},
            pytest.raises(ValueError),
            "'start_time'",
        ),
        # start_time greater than end_time
        (
            {"start_time": 2.0, "end_time": 1.0},
            pytest.raises(ValueError),
            "'start_time'",
        ),
    ],
    ids=[
        "video_not_found",
        "n_frames_zero",
        "n_frames_negative",
        "start_time_negative",
        "end_time_exceeds_duration",
        "start_equals_end",
        "start_greater_than_end",
    ],
)
def test_extract_frames_uniform_invalid_inputs(
    kwargs_override: dict,
    expected_exception: pytest.raises,
    error_fragment: str,
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """extract_frames_uniform raises the expected error for invalid inputs."""
    base_kwargs: dict = {
        "video_path": sample_video["path"],
        "n_frames": 5,
        "output_dir": tmp_path / "frames",
    }
    base_kwargs.update(kwargs_override)

    with expected_exception as excinfo:
        extract_frames_uniform(**base_kwargs)

    assert error_fragment in str(excinfo.value)


def test_extract_frames_uniform_accepts_str_paths(
    sample_video: dict,
    tmp_path: Path,
) -> None:
    """extract_frames_uniform accepts plain strings for path arguments."""
    output_dir = tmp_path / "frames"
    paths = extract_frames_uniform(
        video_path=str(sample_video["path"]),
        n_frames=3,
        output_dir=str(output_dir),
    )
    assert len(paths) == 3
