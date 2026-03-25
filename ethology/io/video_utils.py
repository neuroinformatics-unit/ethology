"""Utility function for extracting video metadata."""

import json
import subprocess
from pathlib import Path


def get_video_specs(video_path: str):
    """Extract metadata from all streams in a video file using ffprobe.

    Parameters
    ----------
    video_path : str
            Path to the video file.

    Returns
    -------
    dict[str, Any]
        Dictionary containing 'duration' (float) and 'streams' (list of dicts).

    Raises
    ------
    FileNotFoundError if the video file does not exist.
    RuntimeError if ffprobe failed to process the file

    Example
    -------
    Get the specifications of a video file

    >>> from ethology.io.video_utils import get_video_specs
    >>> test_file = "path/to/video_file.mp4"
    >>> specs = get_video_specs(test_file)
    >>> print(json.dumps(specs, indent=2))

    """
    # To check whether the file exists
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "stream=index, codec_type, width, height, nb_frames,\
                         r_frame_rate, sample_rate, channels, codec_name",
        "-show_entries",
        "format=duration",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    streams = []
    for s in data.get("streams", []):
        info = {
            "index": s.get("index"),
            "type": s.get("codec_type"),
            "codec": s.get("codec_name"),
        }

        if info["type"] == "video":
            info.update(
                {
                    "width": s.get("width"),
                    "height": s.get("height"),
                    "total_frames": s.get("nb_frames"),
                    "frame_rate": s.get("r_frame_rate"),
                }
            )
        elif info["type"] == "audio":
            info.update(
                {
                    "sample_rate": s.get("sample_rate"),
                    "channels": s.get("channels"),
                }
            )

        streams.append(info)

    return {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "streams": streams,
    }
