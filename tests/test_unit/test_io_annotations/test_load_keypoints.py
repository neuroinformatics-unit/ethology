from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from ethology.io.annotations import load_keypoints


# Mock classes to simulate sleap-io objects
class MockPoint:
    def __init__(
        self,
        x=None,
        y=None,
        visible=None,
        score=None,
        is_visible=None,
        confidence=None,
    ):
        self.x = x
        self.y = y
        self.visible = visible
        if is_visible is not None:
            self.is_visible = is_visible
        self.score = score
        if confidence is not None:
            self.confidence = confidence


class MockInstance:
    def __init__(self, points=None, points_array=None, numpy_func=None):
        self.points = points
        self.points_array = points_array
        self._numpy_func = numpy_func

    def numpy(self):
        if self._numpy_func:
            return self._numpy_func()
        if self.points_array is not None:
            return self.points_array
        return np.array([[p.x, p.y] for p in self.points])


class MockNode:
    def __init__(self, name):
        self.name = name


class MockSkeleton:
    def __init__(self, nodes):
        self.nodes = [MockNode(n) for n in nodes]


class MockVideo:
    def __init__(self, filename=None, path=None, source=None, name=None):
        self.filename = filename
        self.path = path
        self.source = source
        self.name = name


class MockFrame:
    def __init__(
        self,
        frame_idx=None,
        frame_index=None,
        frame_number=None,
        video=None,
        instances=None,
        user_instances=None,
        predicted_instances=None,
    ):
        if frame_idx is not None:
            self.frame_idx = frame_idx
        if frame_index is not None:
            self.frame_index = frame_index
        if frame_number is not None:
            self.frame_number = frame_number
        self.video = video
        self.instances = instances
        self.user_instances = user_instances
        self.predicted_instances = predicted_instances


class MockLabels:
    def __init__(
        self,
        labeled_frames=None,
        frames=None,
        labeled_frames_by_video=None,
        skeletons=None,
        skeleton=None,
    ):
        if labeled_frames is not None:
            self.labeled_frames = labeled_frames
        if frames is not None:
            self.frames = frames
        if labeled_frames_by_video is not None:
            self.labeled_frames_by_video = labeled_frames_by_video
        self.skeletons = skeletons
        self.skeleton = skeleton


def test_require_sleap_io_missing():
    with patch.dict("sys.modules", {"sleap_io": None}):
        with pytest.raises(ModuleNotFoundError, match="sleap-io is required"):
            load_keypoints._require_sleap_io()


def test_get_labeled_frames():
    frames = [1, 2, 3]
    l1 = MockLabels(labeled_frames=frames)
    assert load_keypoints._get_labeled_frames(l1) == frames

    l2 = MockLabels(frames=frames)
    assert load_keypoints._get_labeled_frames(l2) == frames

    l3 = MockLabels(labeled_frames_by_video={"v1": frames})
    assert load_keypoints._get_labeled_frames(l3) == frames

    with pytest.raises(AttributeError, match="Could not find labeled frames"):
        load_keypoints._get_labeled_frames(MockLabels())


def test_get_frame_index():
    assert load_keypoints._get_frame_index(MockFrame(frame_idx=10)) == 10
    assert load_keypoints._get_frame_index(MockFrame(frame_index=11)) == 11
    assert load_keypoints._get_frame_index(MockFrame(frame_number=12)) == 12
    with pytest.raises(AttributeError, match="Could not find frame index"):
        load_keypoints._get_frame_index(MockFrame())


def test_get_video_filename():
    assert load_keypoints._get_video_filename(MockFrame()) is None

    v1 = MockVideo(filename="v1.mp4")
    assert load_keypoints._get_video_filename(MockFrame(video=v1)) == "v1.mp4"

    v2 = MockVideo(path="v2.mp4")
    assert load_keypoints._get_video_filename(MockFrame(video=v2)) == "v2.mp4"

    v3 = MockVideo(source="v3.mp4")
    assert load_keypoints._get_video_filename(MockFrame(video=v3)) == "v3.mp4"

    v4 = MockVideo(name="v4.mp4")
    assert load_keypoints._get_video_filename(MockFrame(video=v4)) == "v4.mp4"

    v5 = MockVideo()
    assert load_keypoints._get_video_filename(MockFrame(video=v5)) is None


def test_get_instances():
    insts = [1, 2]
    assert (
        load_keypoints._get_instances(MockFrame(user_instances=insts)) == insts
    )
    assert load_keypoints._get_instances(MockFrame(instances=insts)) == insts
    assert (
        load_keypoints._get_instances(MockFrame(predicted_instances=insts))
        == insts
    )
    assert load_keypoints._get_instances(MockFrame()) == []
    assert load_keypoints._get_instances(MockFrame(instances=None)) == []


def test_points_from_point_objects():
    p1 = MockPoint(x=10, y=20, visible=True, score=0.9)
    p2 = MockPoint(x=30, y=40, is_visible=False, confidence=0.8)
    p3 = MockPoint(x=None, y=None)  # Missing coords
    p4 = None  # Missing point

    points = [p1, p2, p3, p4]
    coords, conf, vis = load_keypoints._points_from_point_objects(points, 4)

    assert np.allclose(coords[0], [10, 20])
    assert np.allclose(coords[1], [30, 40])
    assert np.isnan(coords[2]).all()
    assert np.isnan(coords[3]).all()

    assert conf[0] == 0.9
    assert conf[1] == 0.8
    assert np.isnan(conf[2])

    assert vis[0] == 1.0
    assert vis[1] == 0.0
    assert np.isnan(vis[2])


def test_points_from_instance():
    # List of points
    p1 = MockPoint(x=1, y=2)
    inst_list = MockInstance(points=[p1])
    coords, _, _ = load_keypoints._points_from_instance(inst_list, 1)
    assert np.allclose(coords, [[1, 2]])

    # Numpy array (n_kp, 2)
    arr_2d = np.array([[1, 2], [3, 4]])
    inst_np = MockInstance(points_array=arr_2d)
    coords, _, _ = load_keypoints._points_from_instance(inst_np, 2)
    assert np.allclose(coords, arr_2d)

    # Numpy array (2, n_kp) -> should transpose
    arr_2d_T = np.array([[1, 3], [2, 4]])
    inst_np_T = MockInstance(points_array=arr_2d_T)
    coords, _, _ = load_keypoints._points_from_instance(inst_np_T, 2)
    assert np.allclose(coords, arr_2d)

    # Numpy array (n_kp, 1, 2)
    arr_3d = np.array([[[1, 2]], [[3, 4]]])
    inst_3d = MockInstance(points_array=arr_3d)
    coords, _, _ = load_keypoints._points_from_instance(inst_3d, 2)
    assert np.allclose(coords, arr_2d)

    # Error case
    with pytest.raises(ValueError, match="Unsupported instance points format"):
        load_keypoints._points_from_instance(
            MockInstance(points_array=np.array([1])), 1
        )


def test_get_skeleton_keypoints():
    nodes = ["head", "tail"]
    sk = MockSkeleton(nodes)
    l1 = MockLabels(skeletons=[sk])
    assert load_keypoints._get_skeleton_keypoints(l1) == nodes

    l2 = MockLabels(skeleton=sk)
    assert load_keypoints._get_skeleton_keypoints(l2) == nodes

    assert load_keypoints._get_skeleton_keypoints(MockLabels()) == []


def test_infer_keypoint_count():
    inst_list = MockInstance(points=[1, 2, 3])
    assert load_keypoints._infer_keypoint_count(inst_list) == 3

    inst_np_2d = MockInstance(points_array=np.zeros((3, 2)))
    assert load_keypoints._infer_keypoint_count(inst_np_2d) == 3

    inst_np_2d_T = MockInstance(points_array=np.zeros((2, 3)))
    assert load_keypoints._infer_keypoint_count(inst_np_2d_T) == 3

    inst_np_3d = MockInstance(points_array=np.zeros((3, 1, 2)))
    assert load_keypoints._infer_keypoint_count(inst_np_3d) == 3

    with pytest.raises(ValueError, match="Could not infer keypoint count"):
        load_keypoints._infer_keypoint_count(MockInstance())


def test_from_single_file_no_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        load_keypoints._from_single_file("path", "INVALID", None)


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file(mock_require):
    mock_sio = MagicMock()
    mock_require.return_value = mock_sio

    # Mock Labels
    p1 = MockPoint(x=10, y=20, visible=True, score=0.9)
    inst = MockInstance(points=[p1])
    frame = MockFrame(
        frame_idx=0, video=MockVideo("vid.mp4"), instances=[inst]
    )
    sk = MockSkeleton(["kp1"])
    labels = MockLabels(labeled_frames=[frame], skeletons=[sk])

    mock_sio.load_file.return_value = labels

    ds = load_keypoints._from_single_file("dummy.slp", "SLEAP", None)

    assert "position" in ds
    assert ds.sizes["image_id"] == 1
    assert ds.sizes["keypoint"] == 1
    assert ds.sizes["id"] == 1
    assert ds.keypoint.values[0] == "kp1"
    assert np.allclose(ds.position.values[0, 0, 0, 0], 10)
    assert np.allclose(ds.position.values[0, 1, 0, 0], 20)
    assert ds.confidence.values[0, 0, 0] == 0.9
    assert ds.visibility.values[0, 0, 0] == 1.0


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file_inference(mock_require):
    mock_sio = MagicMock()
    mock_require.return_value = mock_sio

    # Mock Labels without skeleton but with instances
    p1 = MockPoint(x=10, y=20)
    inst = MockInstance(points=[p1])
    frame = MockFrame(
        frame_idx=0, video=MockVideo("vid.mp4"), instances=[inst]
    )
    labels = MockLabels(labeled_frames=[frame], skeletons=[])

    mock_sio.load_file.return_value = labels

    ds = load_keypoints._from_single_file("dummy.slp", "SLEAP", None)

    assert ds.sizes["keypoint"] == 1
    assert ds.keypoint.values[0] == "keypoint_0"


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file_errors(mock_require):
    mock_sio = MagicMock()
    mock_require.return_value = mock_sio

    # No frames
    mock_sio.load_file.return_value = MockLabels(labeled_frames=[])
    with pytest.raises(ValueError, match="No labeled frames found"):
        load_keypoints._from_single_file("dummy.slp", "SLEAP", None)

    # No instances
    frame = MockFrame(frame_idx=0, video=MockVideo("vid.mp4"), instances=[])
    mock_sio.load_file.return_value = MockLabels(labeled_frames=[frame])
    with pytest.raises(ValueError, match="No instances found"):
        load_keypoints._from_single_file("dummy.slp", "SLEAP", None)


@patch("ethology.io.annotations.load_keypoints._from_single_file")
def test_from_files_multiple(mock_single):
    # Setup two mock datasets with 'space' coord
    ds1 = xr.Dataset(
        {
            "position": (
                ("image_id", "space", "keypoint", "id"),
                np.zeros((1, 2, 1, 1)),
            )
        },
        coords={
            "image_id": [0],
            "keypoint": ["kp1"],
            "id": [0],
            "space": ["x", "y"],
        },
    )
    ds1.attrs = {
        "map_keypoint_to_str": {0: "kp1"},
        "map_image_id_to_filename": {0: "v1_f0"},
        "map_image_id_to_video": {0: "v1"},
        "map_image_id_to_frame_idx": {0: 0},
    }

    ds2 = xr.Dataset(
        {
            "position": (
                ("image_id", "space", "keypoint", "id"),
                np.zeros((1, 2, 1, 1)),
            )
        },
        coords={
            "image_id": [0],
            "keypoint": ["kp1"],
            "id": [0],
            "space": ["x", "y"],
        },
    )
    ds2.attrs = {
        "map_keypoint_to_str": {0: "kp1"},
        "map_image_id_to_filename": {0: "v2_f0"},
        "map_image_id_to_video": {0: "v2"},
        "map_image_id_to_frame_idx": {0: 0},
    }

    mock_single.side_effect = [ds1, ds2]

    ds_out = load_keypoints.from_files(["f1.slp", "f2.slp"])

    assert ds_out.sizes["image_id"] == 2
    assert len(ds_out.attrs["map_image_id_to_filename"]) == 2
    assert ds_out.attrs["map_image_id_to_filename"][0] == "v1_f0"
    assert ds_out.attrs["map_image_id_to_filename"][1] == "v2_f0"


@patch("ethology.io.annotations.load_keypoints._from_single_file")
def test_from_files_mismatch(mock_single):
    ds1 = xr.Dataset(attrs={"map_keypoint_to_str": {0: "kp1"}})
    ds2 = xr.Dataset(attrs={"map_keypoint_to_str": {0: "kp2"}})

    mock_single.side_effect = [ds1, ds2]

    with pytest.raises(ValueError, match="Keypoint labels differ"):
        load_keypoints.from_files(["f1.slp", "f2.slp"])
