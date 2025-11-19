import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import typer
from loguru import logger
from typing_extensions import Annotated

VALID_METHODS = ["uniform"]

app = typer.Typer()


def get_method_function(method: str):
    """Get the function for a given method."""
    if method == "uniform":
        return uniform_frame_extraction
    else:
        raise ValueError(f"Method {method} not supported.")


def uniform_frame_extraction(video_file: Path, n_frames: int) -> list[int]:
    """Extract frames from a video file.

    Parameters
    ----------
    video_file : Path
        The path to the video file.
    n_frames : int
        The number of frames to extract.

    Returns
    -------
    list[int]
        The 0-based frame indices to extract.

    """
    # Get total number of frames in the video
    video = cv2.VideoCapture(str(video_file))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    # Compute (0-indexed) frames to extract
    frames_to_extract = np.linspace(
        0, total_frames - 1, n_frames, endpoint=True, dtype=int
    )
    return frames_to_extract


def compute_frames_to_extract(
    video_dir: Path,
    output_dir: Path,
    method: str,
    method_args: dict,
    video_extension: str = "mp4",
) -> Path:
    """Compute frames to label from all videos in a directory using a given method.

    Parameters
    ----------
    video_dir : Path
        The path to the directory containing the video files.
    output_dir : Path
        The directory to save the list of frames to label.
    method : str
        The method to use to extract the frames.
    method_args : dict
        The arguments to pass to the method.
    video_extension : str
        The extension of the video files. Defaults to "mp4".

    Returns
    -------
    Path
        The path to the metadata file containing the frames to label per video.

    Raises
    ------
    ValueError
        If the method is not supported.
    """
    # Get list of full paths to video files
    if video_dir.is_file():
        video_files = [video_dir]  # .resolve()?
    else:
        video_files = list(video_dir.glob(f"*.{video_extension}"))

        # Check if there are any video files
        if len(video_files) == 0:
            raise ValueError("No video files found in the directory.")

    # Check if the method is supported
    if method not in VALID_METHODS:
        raise ValueError(f"Method {method} not supported.")

    # Compute frames to extract per video
    map_video_to_frame_idcs = {}
    for video_file in video_files:
        frame_idcs = get_method_function(method)(video_file, **method_args)
        map_video_to_frame_idcs[video_file] = frame_idcs

    # Save metadata file to the output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = (
        output_dir / f"{video_dir.name}_{method}_frames2label_{timestamp}.json"
    )
    with open(metadata_file, "w") as f:
        json.dump(map_video_to_frame_idcs, f)


def extract_frames_from_metadata(
    frames_to_label_file: Path,
    output_dir: Path,
) -> Path:
    """Extract frames as specified in the metadata file.

    Parameters
    ----------
    frames_to_label_file : Path
        The path to the metadata file containing the frames to label per video.
        It should hold as keys the full paths to the video files and as values
        the 0-based frame indices to extract.
    output_dir : Path
        The directory to save the extracted frames.

    Returns
    -------
    Path
        The path to the directory containing the extracted frames.
    """
    # Load metadata file
    with open(frames_to_label_file) as f:
        map_video_to_frame_idcs = json.load(f)

    # Create output directory for frames if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop over videos
    for video_file_path, frame_idcs in map_video_to_frame_idcs.items():
        video_capture = cv2.VideoCapture(str(video_file_path))

        if not video_capture.isOpened():
            logger.info(f"Error processing {video_file_path}, skipped....")
            continue

        # Extract frames
        # Should I use ffmpeg instead?
        for frame_idx in frame_idcs:
            # Set video at frame index
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video_capture.read()
            if not success or frame is None:
                logger.error(
                    f"Error reading frame {frame_idx} from {video_file_path}"
                )
                continue

            # Save frame
            frame_file_path = (
                output_dir
                / f"{video_file_path.stem}_frame_{frame_idx:08d}.png"
            )
            img_saved = cv2.imwrite(str(frame_file_path), frame)
            if not img_saved:
                logger.error(f"Error saving frame {frame_file_path}")
                continue

        # Release video capture
        video_capture.release()

    return output_dir


def extract_frames_from_video_dir(
    video_dir: Path | str,
    method: str,
    method_args: dict,
    output_dir: Path | str,
    video_extension: str = "mp4",
) -> Path:
    """Extract frames from a video file."""
    # Convert to Path objects if they are strings
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)

    # Compute frames to label per video
    metadata_file = compute_frames_to_extract(
        video_dir, output_dir, method, method_args, video_extension
    )

    # Extract frames
    frames_dir = extract_frames_from_metadata(metadata_file, output_dir)

    return frames_dir


@app.command()
def app_wrapper(
    video_dir: Annotated[Path | str, typer.Option()],
    method: Annotated[str, typer.Option()],
    method_args: Annotated[
        list[str], typer.Option(..., help="key=value pairs")
    ],
    output_dir: Annotated[Path | str, typer.Option()],
):
    """Extract frames from a video directory using a given method.

    Parameters
    ----------
    video_dir : Path | str
        The path to the directory containing the video files.
    method : str
        The method to use to extract the frames.
    method_args : list[str]
        The arguments to pass to the method.
    output_dir : Path | str
        The directory to save the extracted frames.

    """
    typer.echo(f"Extracting frames from {video_dir} using {method} method...")

    # Parse method arguments
    method_args = {
        k: v
        for k, v in [key_val_str.split("=") for key_val_str in method_args]
    }

    # Extract frames
    extract_frames_from_video_dir(video_dir, method, method_args, output_dir)


if __name__ == "__main__":
    app_wrapper()
