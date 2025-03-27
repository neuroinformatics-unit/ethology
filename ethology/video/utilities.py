import os

import ffmpeg


def compress_video(
    input_path: str, output_path: str, crf: int = 23, preset: str = "medium"
) -> None:
    """Compress a video by applying a reasonable compression.

    Args:
        input_path: Path to the input video.
        output_path: Path for the output (compressed) video.
        crf: Constant Rate Factor (lower values mean better quality; typical range is 18-28).
        preset: ffmpeg preset for compression speed/quality trade-off.

    """
    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, vcodec="libx264", crf=crf, preset=preset)
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Error compressing video: {e.stderr.decode()}"
        ) from e


def get_video_specs(input_path: str) -> dict:
    """Get main video specifications using ffprobe via ffmpeg-python.

    Returns a dictionary with keys:
      - frame_rate: Frames per second.
      - codec: Video codec name.
      - width: Frame width.
      - height: Frame height.
      - nb_frames: Number of frames (if available).
      - duration: Duration in seconds.
    """
    try:
        probe = ffmpeg.probe(input_path)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Error probing video: {e.stderr.decode()}") from e

    # Find the first video stream
    video_stream = next(
        (
            stream
            for stream in probe.get("streams", [])
            if stream.get("codec_type") == "video"
        ),
        None,
    )
    if not video_stream:
        raise ValueError("No video stream found.")

    # Compute the frame rate from r_frame_rate (e.g., "30000/1001")
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0
    except Exception:
        fps = 0

    # Get duration: sometimes in the video stream, otherwise in format.
    duration = float(video_stream.get("duration", 0))
    if duration == 0:
        duration = float(probe.get("format", {}).get("duration", 0))

    specs = {
        "frame_rate": fps,
        "codec": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "nb_frames": int(video_stream.get("nb_frames", 0)),
        "duration": duration,
    }
    return specs


def print_video_specs(input_path: str) -> None:
    """Print key video specifications to the terminal."""
    specs = get_video_specs(input_path)
    print("Video Specifications:")
    print(f"  Codec: {specs.get('codec', 'Unknown')}")
    print(f"  Frame Rate: {specs.get('frame_rate', 'Unknown'):.2f} fps")
    print(f"  Duration: {specs.get('duration', 'Unknown')} seconds")
    print(f"  Number of Frames: {specs.get('nb_frames', 'Unknown')}")
    if specs.get("width") and specs.get("height"):
        print(f"  Frame Size: {specs.get('width')}x{specs.get('height')}")


def extract_frames(
    input_path: str,
    output_dir: str,
    frame_numbers: list[int] | None = None,
    time_range: tuple[int | float, int | float] | None = None,
    all_frames: bool = False,
) -> None:
    """Extract frames from a video.

    Supports three modes:
      - all_frames=True: Extract every frame.
      - frame_numbers: Extract frames based on frame indices.
      - time_range: Extract frames from a time interval (start, end in seconds).

    Args:
        input_path: Path to the input video.
        output_dir: Directory where frames will be saved.
        frame_numbers: List of frame indices to extract.
        time_range: Tuple (start, end) in seconds to extract frames.
        all_frames: If True, extract all frames.

    """
    os.makedirs(output_dir, exist_ok=True)

    if all_frames:
        # Use a pattern to output every frame (e.g., frame_000001.jpg, frame_000002.jpg, etc.)
        output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
        try:
            (
                ffmpeg.input(input_path)
                .output(output_pattern)
                .run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Error extracting all frames: {e.stderr.decode()}"
            ) from e

    elif frame_numbers:
        # Calculate timestamps from frame numbers based on frame rate.
        specs = get_video_specs(input_path)
        fps = specs.get("frame_rate", 30)  # fallback fps=30 if not available
        for frame in frame_numbers:
            time_sec = frame / fps
            output_file = os.path.join(output_dir, f"frame_{frame:06d}.jpg")
            try:
                (
                    ffmpeg.input(input_path, ss=time_sec)
                    .output(output_file, vframes=1)
                    .run(quiet=True, overwrite_output=True)
                )
            except ffmpeg.Error as e:
                raise RuntimeError(
                    f"Error extracting frame {frame}: {e.stderr.decode()}"
                ) from e

    elif time_range:
        start, end = time_range
        duration = end - start
        output_pattern = os.path.join(output_dir, "frame_%06d.jpg")
        try:
            (
                ffmpeg.input(input_path, ss=start, t=duration)
                .output(output_pattern)
                .run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Error extracting frames in time range: {e.stderr.decode()}"
            ) from e
    else:
        raise ValueError(
            "Must specify either all_frames, frame_numbers, or time_range."
        )


def combine_images_to_video(
    image_pattern: str, output_video: str, fps: int = 30
) -> None:
    """Combine a sequence of images into a video.

    Args:
        image_pattern: Pattern for input images (e.g., 'frames/frame_%06d.jpg').
        output_video: Path for the output video.
        fps: Frame rate for the resulting video.

    """
    try:
        (
            ffmpeg.input(image_pattern, framerate=fps)
            .output(output_video, vcodec="libx264", pix_fmt="yuv420p")
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Error combining images into video: {e.stderr.decode()}"
        ) from e


def crop_video_time(
    input_path: str,
    output_path: str,
    start: int | float | None = None,
    end: int | float | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> None:
    """Crop a video in time. You can specify the crop interval either in seconds (start, end)
    or in frame numbers (start_frame, end_frame).

    Args:
        input_path: Path to the input video.
        output_path: Path for the cropped output video.
        start: Start time in seconds.
        end: End time in seconds.
        start_frame: Start frame index.
        end_frame: End frame index.

    """
    if start_frame is not None and end_frame is not None:
        specs = get_video_specs(input_path)
        fps = specs.get("frame_rate", 30)
        start = start_frame / fps
        end = end_frame / fps

    if start is None or end is None:
        raise ValueError(
            "Must specify a valid time range either in seconds or in frame numbers."
        )

    duration = end - start
    try:
        (
            ffmpeg.input(input_path, ss=start, t=duration)
            .output(output_path)
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Error cropping video in time: {e.stderr.decode()}"
        ) from e


def crop_video_spatial(
    input_path: str, output_path: str, x: int, y: int, width: int, height: int
) -> None:
    """Crop a video spatially (i.e., crop every frame).

    Args:
        input_path: Path to the input video.
        output_path: Path for the output (cropped) video.
        x: The x-coordinate of the top-left corner of the crop rectangle.
        y: The y-coordinate of the top-left corner of the crop rectangle.
        width: The width of the crop rectangle.
        height: The height of the crop rectangle.

    """
    try:
        (
            ffmpeg.input(input_path)
            .crop(x, y, width, height)
            .output(output_path)
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Error cropping video spatially: {e.stderr.decode()}"
        ) from e
