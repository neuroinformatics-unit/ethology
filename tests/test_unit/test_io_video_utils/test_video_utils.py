import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ethology.io.video_utils import compress_video, get_video_specs


@pytest.fixture
def valid_get_video_specs():
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


def test_get_video_specs_valid_file(valid_video_file, valid_get_video_specs):
    """Test function returns correct structure for valid video."""
    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=valid_get_video_specs,
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
    valid_video_file, valid_get_video_specs
):
    """Test video streams contain expected metadata."""
    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=valid_get_video_specs,
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
    valid_video_file, valid_get_video_specs
):
    """Test audio streams contain expected metadata."""
    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=valid_get_video_specs,
    ):
        result = get_video_specs(str(valid_video_file))

    audio_streams = [s for s in result["streams"] if s["type"] == "audio"]
    assert len(audio_streams) > 0

    stream = audio_streams[0]
    assert stream["sample_rate"] == "48000"
    assert stream["channels"] == 2


## Video Compression


@pytest.fixture
def valid_compression_run():
    return subprocess.CompletedProcess(
        returncode=0, cmd=["ffmpeg"], stderr=b"", stdout=""
    )


def test_compress_video_file_not_found(tmp_path):
    """Test that FileNotFoundError is raised if input file does not exist."""
    input_file = tmp_path / "non_existent.mp4"
    output_file = tmp_path / "output.mp4"

    with pytest.raises(
        FileNotFoundError, match=f"Video file not found: {input_file}"
    ):
        compress_video(str(input_file), str(output_file))


def test_compress_video_success(valid_video_file, tmp_path):
    """Test successful compression when input file exists."""
    output_file = tmp_path / "output.mp4"

    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=valid_compression_run,
    ) as mock_run:
        result = compress_video(str(valid_video_file), str(output_file))

        assert result is True  # Check if compression worked

        mock_run.assert_called_once()

        # Checking if valid args were passed to mock_run
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert str(valid_video_file) in call_args
        assert str(output_file) in call_args


def test_compress_video_failure(valid_video_file, tmp_path):
    """Test handling of FFmpeg failure (CalledProcessError)."""
    output_file = tmp_path / "output.mp4"

    with patch("ethology.io.video_utils.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["ffmpeg", "..."], stderr=b"Error: Invalid data"
        )

        result = compress_video(str(valid_video_file), str(output_file))

        assert result is False


def test_compress_video_overwrite_false(valid_video_file, tmp_path):
    """Test that -y flag is omitted when overwrite=False."""
    output_file = tmp_path / "output.mp4"

    with patch(
        "ethology.io.video_utils.subprocess.run",
        return_value=valid_compression_run,
    ) as mock_run:
        compress_video(
            str(valid_video_file), str(output_file), overwrite=False
        )

        call_args = mock_run.call_args[0][0]
        assert "-y" not in call_args
