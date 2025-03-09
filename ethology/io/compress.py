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

    Args:
        video_path (str): local video filepath to be trimmed
        saveout_path (str): local filepath for saving file to
        crf (int, optional): Compression rate factor. Defaults to 23.
        overwrite_output (bool, optional): Whether to overwrite an
            existing video on saveout_path.

    """
    # check_ffmpeg_installed()
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
            capture_stdout=True,
            capture_stderr=True,
        )
    except ffmpeg._run.Error as e:
        raise RuntimeError(
            f"FFmpeg Error: \n {e.stderr.decode()
                                if e.stderr else "No stderr"}"
        ) from e
