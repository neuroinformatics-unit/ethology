import json
from unittest.mock import MagicMock, patch

import pytest

from ethology.io.video_utils import get_video_specs


@pytest.fixture
def mock_ffprobe_success():
    """Mock successful ffprobe execution."""
    mock_data = {
        "format": {"duration": "123.456"},
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "nb_frames": 3720,
                "r_frame_rate": "30/1",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
    }
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(mock_data)
    mock_result.stderr = ""
    return mock_result


@pytest.fixture
def valid_video_file(tmp_path):
    """Create a temporary video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    return video_path


def test_get_video_specs_valid_file(valid_video_file, mock_ffprobe_success):
    """Test function returns correct structure for valid video."""
    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=mock_ffprobe_success,
    ):
        result = get_video_specs(str(valid_video_file))

    assert isinstance(result, dict)
    assert "duration" in result
    assert "streams" in result
    assert isinstance(result["duration"], float)
    assert isinstance(result["streams"], list)


def test_get_video_specs_missing_file():
    """Test function raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError) as excinfo:
        get_video_specs("nonexistent_file.mp4")

    assert "Video file not found" in str(excinfo.value)


def test_get_video_specs_ffprobe_failure(valid_video_file):
    """Test function raises RuntimeError when ffprobe fails."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "ffprobe error"

    with (
        patch(
            "ethology.io.video_utils.subprocess.run", return_value=mock_result
        ),
        pytest.raises(RuntimeError) as excinfo,
    ):
        get_video_specs(str(valid_video_file))

    assert "ffprobe failed" in str(excinfo.value)


def test_video_stream_has_required_fields(
    valid_video_file, mock_ffprobe_success
):
    """Test video streams contain expected metadata."""
    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=mock_ffprobe_success,
    ):
        result = get_video_specs(str(valid_video_file))

    video_streams = [s for s in result["streams"] if s["type"] == "video"]
    assert len(video_streams) > 0

    stream = video_streams[0]
    assert stream["width"] == 1920
    assert stream["height"] == 1080
    assert stream["total_frames"] == 3720
    assert stream["frame_rate"] == "30/1"


def test_audio_stream_has_required_fields(
    valid_video_file, mock_ffprobe_success
):
    """Test audio streams contain expected metadata."""
    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=mock_ffprobe_success,
    ):
        result = get_video_specs(str(valid_video_file))

    audio_streams = [s for s in result["streams"] if s["type"] == "audio"]
    assert len(audio_streams) > 0

    stream = audio_streams[0]
    assert stream["sample_rate"] == "48000"
    assert stream["channels"] == 2
