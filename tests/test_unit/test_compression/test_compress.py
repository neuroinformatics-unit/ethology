import os
import tempfile

import pytest
import sleap_io as sio
from movement import sample_data

from ethology.io.compress import compress_video_h264


@pytest.fixture
def test_video():
    """Fetches a sample dataset and returns video information."""
    ds = sample_data.fetch_dataset(
        "SLEAP_three-mice_Aeon_proofread.analysis.h5", with_video=True
    )
    video = sio.load_video(ds.video_path)
    return video


def test_compress_video_h264(test_video):
    """Test if compress_video_h264 correctly processes the video."""
    video = test_video

    # Generate temporary output filename
    with tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=True
    ) as temp_output:
        output_filename = temp_output.name

    try:
        compress_video_h264(video.filename, output_filename, crf=28)

        # Load the output video
        output_video = sio.load_video(output_filename)

        # Assert that the video frame count matches
        assert len(video) == len(output_video)

    finally:
        # Clean up the output file after test completion
        if os.path.exists(output_filename):
            os.remove(output_filename)
