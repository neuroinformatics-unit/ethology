"""Load keypoints annotations into ``ethology``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from ethology.validators.annotations import ValidKeypointsAnnotationsDataset
from ethology.validators.utils import _check_output


def _require_sleap_io():
    try:
        import sleap_io as sio  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sleap-io is required for keypoints IO. "
            "Install it with `pip install sleap-io`."
        ) from exc
    return sio


def _get_labeled_frames(labels: Any) -> list[Any]:
    if hasattr(labels, "labeled_frames"):
        return list(labels.labeled_frames)
    if hasattr(labels, "frames"):
        return list(labels.frames)
    if hasattr(labels, "labeled_frames_by_video"):
        frames: list[Any] = []
        for frames_list in labels.labeled_frames_by_video.values():
            frames.extend(list(frames_list))
        return frames
    raise AttributeError(
        "Could not find labeled frames on sleap Labels object."
    )


def _get_frame_index(frame: Any) -> int:
    for attr in ["frame_idx", "frame_index", "frame_number"]:
        if hasattr(frame, attr):
            return int(getattr(frame, attr))
    raise AttributeError("Could not find frame index on labeled frame.")


def _get_video_filename(frame: Any) -> str | None:
    if not hasattr(frame, "video") or frame.video is None:
        return None
    video = frame.video
    for attr in ["filename", "path", "source", "name"]:
        if hasattr(video, attr):
            value = getattr(video, attr)
            if value is None:
                continue
            return str(value)
    return None


def _get_instances(frame: Any) -> list[Any]:
    for attr in ["user_instances", "instances", "predicted_instances"]:
        if hasattr(frame, attr):
            instances = getattr(frame, attr)
            if instances is None:
                continue
            instances_list = list(instances)
            if instances_list:
                return instances_list
    return []


def _points_from_point_objects(
    points: list[Any], n_keypoints: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    coords = np.full((n_keypoints, 2), np.nan, dtype=np.float64)
    confidence = np.full((n_keypoints,), np.nan, dtype=np.float64)
    visibility = np.full((n_keypoints,), np.nan, dtype=np.float64)
    for idx, point in enumerate(points):
        if point is None or idx >= n_keypoints:
            continue
        x = getattr(point, "x", None)
        y = getattr(point, "y", None)
        if x is None or y is None:
            continue
        visible = getattr(point, "visible", None)
        if visible is None:
            visible = getattr(point, "is_visible", None)
        if visible is False:
            visibility[idx] = 0.0
            continue
        if visible is True:
            visibility[idx] = 1.0
        coords[idx] = [float(x), float(y)]
        confidence[idx] = getattr(
            point,
            "score",
            getattr(point, "confidence", np.nan),
        )
    return coords, confidence, visibility


def _points_from_instance(
    instance: Any, n_keypoints: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    for attr in ["numpy", "to_numpy", "points_array", "points"]:
        if not hasattr(instance, attr):
            continue
        value = getattr(instance, attr)
        data = value() if callable(value) else value
        if isinstance(data, list):
            return _points_from_point_objects(data, n_keypoints)
        if isinstance(data, np.ndarray):
            arr = data.astype(np.float64, copy=False)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                if arr.shape[0] == n_keypoints:
                    return arr[:, :2], None, None
                if arr.shape[1] == n_keypoints and arr.shape[0] >= 2:
                    return arr[:2, :].T, None, None
            if arr.ndim == 3 and arr.shape[-1] >= 2:
                # Some formats store (n_keypoints, 1, 2)
                arr = arr.reshape(arr.shape[0], -1)
                if arr.shape[0] == n_keypoints:
                    return arr[:, :2], None, None
    raise ValueError(
        "Unsupported instance points format in sleap Labels object."
    )


def _get_skeleton_keypoints(labels: Any) -> list[str]:
    if hasattr(labels, "skeletons") and labels.skeletons:
        skeleton = labels.skeletons[0]
        if hasattr(skeleton, "nodes"):
            return [node.name for node in skeleton.nodes]
    if hasattr(labels, "skeleton"):
        skeleton = labels.skeleton
        if hasattr(skeleton, "nodes"):
            return [node.name for node in skeleton.nodes]
    return []


def _infer_keypoint_count(instance: Any) -> int:
    for attr in ["numpy", "to_numpy", "points_array", "points"]:
        if not hasattr(instance, attr):
            continue
        value = getattr(instance, attr)
        data = value() if callable(value) else value
        if isinstance(data, list):
            return len(data)
        if isinstance(data, np.ndarray):
            arr = data
            if arr.ndim == 2:
                if arr.shape[1] == 2:
                    return arr.shape[0]
                if arr.shape[0] == 2:
                    return arr.shape[1]
                return arr.shape[0]
            if arr.ndim == 3:
                return arr.shape[0]
    raise ValueError("Could not infer keypoint count from instance.")


def _prepare_frame_records(labels: Any) -> list[dict[str, Any]]:
    frame_records = []
    for frame in _get_labeled_frames(labels):
        frame_idx = _get_frame_index(frame)
        video_filename = _get_video_filename(frame)
        frame_records.append(
            {
                "frame": frame,
                "frame_idx": frame_idx,
                "video_filename": video_filename,
            }
        )
    frame_records.sort(
        key=lambda r: (
            r["video_filename"] or "",
            r["frame_idx"],
        )
    )
    return frame_records


def _frame_label(video_filename: str | None, frame_idx: int) -> str:
    if video_filename:
        return f"{video_filename}::frame_{frame_idx}"
    return f"frame_{frame_idx}"


def _from_single_file(
    file_path: Path | str,
    format: Literal["SLEAP"],
    images_dirs: Path | str | list[Path | str] | None,
) -> xr.Dataset:
    if format != "SLEAP":
        raise ValueError(f"Unsupported format: {format}")

    sio = _require_sleap_io()
    labels = sio.load_file(file_path)
    keypoint_names = _get_skeleton_keypoints(labels)

    frame_records = _prepare_frame_records(labels)
    if not frame_records:
        raise ValueError("No labeled frames found in keypoints file.")

    max_instances = 0
    for record in frame_records:
        instances = _get_instances(record["frame"])
        max_instances = max(max_instances, len(instances))

    if max_instances == 0:
        raise ValueError("No instances found in keypoints file.")

    if not keypoint_names:
        # Fallback: infer number of keypoints from the first instance
        first_instances = _get_instances(frame_records[0]["frame"])
        if not first_instances:
            raise ValueError("No instances found to infer keypoints.")
        n_keypoints = _infer_keypoint_count(first_instances[0])
        keypoint_names = [f"keypoint_{i}" for i in range(n_keypoints)]

    n_keypoints = len(keypoint_names)
    n_frames = len(frame_records)

    position = np.full(
        (n_frames, 2, n_keypoints, max_instances),
        np.nan,
        dtype=np.float64,
    )
    confidence = np.full(
        (n_frames, n_keypoints, max_instances),
        np.nan,
        dtype=np.float64,
    )
    visibility = np.full(
        (n_frames, n_keypoints, max_instances),
        np.nan,
        dtype=np.float64,
    )

    map_image_id_to_filename: dict[int, str] = {}
    map_image_id_to_video: dict[int, str] = {}
    map_image_id_to_frame_idx: dict[int, int] = {}

    for image_id, record in enumerate(frame_records):
        frame = record["frame"]
        frame_idx = record["frame_idx"]
        video_filename = record["video_filename"]
        map_image_id_to_filename[image_id] = _frame_label(
            video_filename, frame_idx
        )
        if video_filename:
            map_image_id_to_video[image_id] = video_filename
        map_image_id_to_frame_idx[image_id] = frame_idx

        for inst_idx, instance in enumerate(_get_instances(frame)):
            coords, conf, vis = _points_from_instance(instance, n_keypoints)
            if coords.shape[0] != n_keypoints:
                raise ValueError(
                    "Instance keypoints do not match skeleton definition."
                )
            position[image_id, :, :, inst_idx] = coords.T
            if conf is not None:
                confidence[image_id, :, inst_idx] = conf
            if vis is not None:
                visibility[image_id, :, inst_idx] = vis

    ds = xr.Dataset(
        data_vars={
            "position": (
                ["image_id", "space", "keypoint", "id"],
                position,
            ),
        },
        coords={
            "image_id": np.arange(n_frames),
            "space": ["x", "y"],
            "keypoint": keypoint_names,
            "id": np.arange(max_instances),
        },
    )

    if np.isfinite(confidence).any():
        ds["confidence"] = (
            ["image_id", "keypoint", "id"],
            confidence,
        )
    if np.isfinite(visibility).any():
        ds["visibility"] = (
            ["image_id", "keypoint", "id"],
            visibility,
        )

    ds.attrs = {
        "annotation_files": file_path,
        "annotation_format": format,
        "images_directories": images_dirs,
        "map_keypoint_to_str": dict(enumerate(keypoint_names)),
        "map_image_id_to_filename": map_image_id_to_filename,
        "map_image_id_to_video": map_image_id_to_video,
        "map_image_id_to_frame_idx": map_image_id_to_frame_idx,
    }
    return ds


@_check_output(ValidKeypointsAnnotationsDataset)
def from_files(
    file_paths: Path | str | list[Path | str],
    format: Literal["SLEAP"] = "SLEAP",
    images_dirs: Path | str | list[Path | str] | None = None,
) -> xr.Dataset:
    """Load an ``ethology`` keypoints annotations dataset from a file.

    Parameters
    ----------
    file_paths : pathlib.Path | str | list[pathlib.Path | str]
        Path or list of paths to the input keypoints annotation files.
    format : {"SLEAP"}
        Format of the input annotation files. Currently only "SLEAP".
    images_dirs : pathlib.Path | str | list[pathlib.Path | str], optional
        Path or list of paths to the directories containing the images the
        annotations refer to. The paths are added to the dataset attributes.

    Returns
    -------
    xarray.Dataset
        A valid keypoints annotations dataset with dimensions
        `image_id`, `space`, `keypoint`, `id` and data variable `position`.

    """
    if isinstance(file_paths, list):
        datasets = []
        map_keypoint_to_str = None
        image_id_offset = 0
        for path in file_paths:
            ds = _from_single_file(path, format=format, images_dirs=images_dirs)
            if map_keypoint_to_str is None:
                map_keypoint_to_str = ds.attrs.get("map_keypoint_to_str")
            elif map_keypoint_to_str != ds.attrs.get("map_keypoint_to_str"):
                raise ValueError(
                    "Keypoint labels differ across input files; "
                    "cannot merge datasets."
                )

            ds = ds.assign_coords(
                image_id=ds.image_id + image_id_offset
            )
            # Update mapping attrs to new image_id range
            map_image_id_to_filename = {}
            map_image_id_to_video = {}
            map_image_id_to_frame_idx = {}
            for old_id in ds.attrs["map_image_id_to_filename"].keys():
                new_id = int(old_id) + image_id_offset
                map_image_id_to_filename[new_id] = ds.attrs[
                    "map_image_id_to_filename"
                ][old_id]
                if old_id in ds.attrs.get("map_image_id_to_video", {}):
                    map_image_id_to_video[new_id] = ds.attrs[
                        "map_image_id_to_video"
                    ][old_id]
                map_image_id_to_frame_idx[new_id] = ds.attrs[
                    "map_image_id_to_frame_idx"
                ][old_id]
            ds.attrs["map_image_id_to_filename"] = map_image_id_to_filename
            ds.attrs["map_image_id_to_video"] = map_image_id_to_video
            ds.attrs["map_image_id_to_frame_idx"] = map_image_id_to_frame_idx

            datasets.append(ds)
            image_id_offset += ds.sizes["image_id"]

        ds_all = xr.concat(datasets, dim="image_id")
        ds_all.attrs = {
            "annotation_files": file_paths,
            "annotation_format": format,
            "images_directories": images_dirs,
            "map_keypoint_to_str": map_keypoint_to_str or {},
            "map_image_id_to_filename": {
                k: v
                for ds in datasets
                for k, v in ds.attrs.get("map_image_id_to_filename", {}).items()
            },
            "map_image_id_to_video": {
                k: v
                for ds in datasets
                for k, v in ds.attrs.get("map_image_id_to_video", {}).items()
            },
            "map_image_id_to_frame_idx": {
                k: v
                for ds in datasets
                for k, v in ds.attrs.get("map_image_id_to_frame_idx", {}).items()
            },
        }
        return ds_all

    return _from_single_file(file_paths, format=format, images_dirs=images_dirs)
