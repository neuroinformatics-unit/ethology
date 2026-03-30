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


def compress_video(
    input_path: str,
    output_path: str,
    crf: int = 23,
    preset: str = "superfast",
    overwrite: bool = True,
):
    """Compress video using H.264 codec with specified quality settings.

    Parameters
    ----------
    input_path : str
            Path to the input video file.

    output_path : str
            Path where the compressed video file will be saved.

    crf : int, optional
            Constant Rate Factor determining the quality and bitrate.
            Lower values yield higher quality and larger file sizes
            (range 0-51, typical 18-28).
            Default is 23.

    preset : str, optional
            The encoding speed preset. Faster presets result in larger files
            but quicker encoding. Options include 'ultrafast', 'superfast',
            'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower',
            'veryslow'.
            Default is 'superfast'.

    overwrite : bool, optional
            If True, overwrite the output file if it already exists.
            Default is True.

    Returns
    -------
    bool
            True if successful, False if an error occurred

    Raises
    ------
            FileNotFoundError if the video file does not exist.

    Example:
    --------
    >>> from ethology.io.video_utils import compress_video
    >>> compress_video("input.mp4", "output.mp4")
           True
    >>> compress_video("input.mp4", "output.mp4", crf=20, preset="medium")
           True

    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-progress",
        "pipe:1",
        str(output_path),
    ]

    cmd = [
        arg for arg in cmd if arg
    ]  # Filter out empty args, say in case of overwrite=False

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("File compressed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        print("File compression failed.")
        return False
