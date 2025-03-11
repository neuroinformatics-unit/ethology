import os
import shutil

import ffmpeg


class FFmpegNotFoundError(Exception):
    """Custom exception for missing ffmpeg installation."""

    pass


def check_ffmpeg_installed():
    """Check if ffmpeg is installed by looking for the executable."""
    if shutil.which("ffmpeg") is None:
        raise FFmpegNotFoundError(
            "FFmpeg is not installed or not found in system PATH."
        )
    else:
        return True


def compress_video_h264(
    video_path: str,
    saveout_path: str,
    crf: int = 23,
    overwrite_output: bool = False,
) -> None:
    """Compresses a local video, and saves out.

    Uses SLEAP's recommended settings as defaults.

    Args:
        video_path (str): local video filepath to be trimmed
        saveout_path (str): local filepath for saving file to
        crf (int, optional): Compression rate factor. Defaults to 23, higher
            means more compression.
        overwrite_output (bool, optional): Whether to overwrite an
            existing video on saveout_path.

    """
    # Pre-ffmpeg checks to handle errors better
    check_ffmpeg_installed()

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Input file specified: {video_path} not " "found."
        )

    if os.path.isfile(saveout_path) and not overwrite_output:
        raise FileExistsError(
            f"Output file specified: {video_path} already "
            "exists. Consider changing overwrite_output to True."
        )

    try:
        input = ffmpeg.input(video_path)
        output = input.output(
            saveout_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="superfast",
            crf=crf,
        )
        output.run(
            overwrite_output=overwrite_output,
        )
    # Catch other ffmpeg errors
    except ffmpeg._run.Error as e:
        raise RuntimeError(
            f"FFmpeg Error: \n {e.stderr.decode()
                                if e.stderr else "No stderr"}"
        ) from e
