"""Save ``ethology`` keypoints annotations datasets to various formats."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from ethology.validators.annotations import ValidKeypointsAnnotationsDataset
from ethology.validators.utils import _check_input


def _require_sleap_io():
    try:
        import sleap_io as sio  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "sleap-io is required for keypoints IO. "
            "Install it with `pip install sleap-io`."
        ) from exc
    return sio


def _get_keypoint_names(ds: xr.Dataset) -> list[str]:
    if "map_keypoint_to_str" in ds.attrs:
        mapping = ds.attrs["map_keypoint_to_str"]
        if isinstance(mapping, dict) and mapping:
            return [mapping[i] for i in range(len(mapping))]
    return [str(kp) for kp in ds.keypoint.values]


def _get_image_id_maps(
    ds: xr.Dataset,
) -> tuple[dict[int, str], dict[int, str], dict[int, int]]:
    map_image_id_to_filename = ds.attrs.get("map_image_id_to_filename", {})
    map_image_id_to_video = ds.attrs.get("map_image_id_to_video", {})
    map_image_id_to_frame_idx = ds.attrs.get("map_image_id_to_frame_idx", {})
    return (
        map_image_id_to_filename,
        map_image_id_to_video,
        map_image_id_to_frame_idx,
    )


def _build_sleap_objects(ds: xr.Dataset) -> Any:  # noqa: C901
    sio = _require_sleap_io()
    keypoint_names = _get_keypoint_names(ds)

    node_cls = getattr(sio, "Node", None)
    skeleton_cls = getattr(sio, "Skeleton", None)
    labeled_frame_cls = getattr(sio, "LabeledFrame", None)
    instance_cls = getattr(sio, "Instance", None)
    video_cls = getattr(sio, "Video", None)
    point_cls = getattr(sio, "Point", None)

    if not all([skeleton_cls, labeled_frame_cls, instance_cls, video_cls]):
        raise AttributeError(
            "sleap-io is missing required classes for saving Labels."
        )

    # Type assertions after None check
    assert skeleton_cls is not None
    assert labeled_frame_cls is not None
    assert instance_cls is not None
    assert video_cls is not None

    nodes = (
        [node_cls(name=name) for name in keypoint_names]
        if node_cls is not None
        else keypoint_names
    )
    try:
        skeleton = skeleton_cls(nodes=nodes, edges=[])
    except TypeError:
        skeleton = skeleton_cls(nodes=nodes)

    (
        map_image_id_to_filename,
        map_image_id_to_video,
        map_image_id_to_frame_idx,
    ) = _get_image_id_maps(ds)  # noqa: E501

    videos: dict[str, Any] = {}
    labeled_frames = []
    confidence = ds.get("confidence")
    visibility = ds.get("visibility")

    for image_id in ds.image_id.values:
        image_id_int = int(image_id)
        video_filename = map_image_id_to_video.get(
            image_id_int,
            map_image_id_to_filename.get(image_id_int, ""),
        )
        if not video_filename:
            raise ValueError(
                "Missing video or filename information in dataset attrs."
            )

        if video_filename not in videos:
            try:
                video = video_cls.from_filename(video_filename)  # type: ignore
            except AttributeError:
                try:
                    video = video_cls(filename=video_filename)  # type: ignore
                except TypeError:
                    video = video_cls(video_filename)  # type: ignore
            videos[video_filename] = video
        else:
            video = videos[video_filename]

        frame_idx = map_image_id_to_frame_idx.get(image_id_int, image_id_int)

        try:
            labeled_frame = labeled_frame_cls(video=video, frame_idx=frame_idx)  # type: ignore
        except TypeError:
            labeled_frame = labeled_frame_cls(video, frame_idx)  # type: ignore

        instances = []
        for inst_id in ds.id.values:
            coords = ds.position.sel(image_id=image_id, id=inst_id).values
            if np.isnan(coords).all():
                continue
            points: list[Any] = []
            for kp_idx, _name in enumerate(keypoint_names):
                x, y = coords[:, kp_idx]
                if np.isnan(x) or np.isnan(y):
                    points.append(None)
                    continue
                score = None
                if confidence is not None:
                    score = float(
                        confidence.sel(image_id=image_id, id=inst_id).values[
                            kp_idx
                        ]
                    )
                visible = None
                if visibility is not None:
                    visible = float(
                        visibility.sel(image_id=image_id, id=inst_id).values[
                            kp_idx
                        ]
                    )
                if point_cls is not None:
                    kwargs = {}
                    if score is not None and not np.isnan(score):
                        kwargs["score"] = score
                    if visible is not None and not np.isnan(visible):
                        kwargs["visible"] = bool(int(visible))
                    point = point_cls(x=float(x), y=float(y), **kwargs)
                    points.append(point)
                else:
                    points.append([float(x), float(y)])

            try:
                instance = instance_cls(points=points, skeleton=skeleton)  # type: ignore
            except TypeError:
                instance = instance_cls(points, skeleton)  # type: ignore
            instances.append(instance)

        if instances:
            labeled_frame.instances = instances
            labeled_frames.append(labeled_frame)

    try:
        labels = sio.Labels(
            labeled_frames=labeled_frames, skeletons=[skeleton]
        )  # type: ignore
    except TypeError:
        labels = sio.Labels(labeled_frames)  # type: ignore
        if hasattr(labels, "skeletons"):
            labels.skeletons = [skeleton]
    return labels


@_check_input(validator=ValidKeypointsAnnotationsDataset)
def to_file(
    dataset: xr.Dataset,
    output_filepath: str | Path,
    format: Literal["SLEAP"] = "SLEAP",
) -> str | Path:
    """Save an ``ethology`` keypoints annotations dataset to a file.

    Parameters
    ----------
    dataset : xarray.Dataset
        Keypoints annotations xarray dataset.
    output_filepath : str or pathlib.Path
        Path for the output file.
    format : {"SLEAP"}
        Format of the output file.

    Returns
    -------
    str or pathlib.Path
        Path for the output file.

    """
    if format != "SLEAP":
        raise ValueError(f"Unsupported format: {format}")

    sio = _require_sleap_io()
    labels = _build_sleap_objects(dataset)
    try:
        sio.save_file(labels, output_filepath)
    except TypeError:
        sio.save_file(output_filepath, labels)
    return output_filepath
