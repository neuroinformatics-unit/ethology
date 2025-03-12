"""Utilities for reading videos, drawing shapes on frames.

    and visualizing tracks.

This module provides helper functions for reading videos, drawing circles and
lines, blending images, and a `Visualizer` class to overlay tracked points on
video frames.
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import cm
from PIL import Image, ImageDraw


def read_video_from_path(path):
    """Read a video from the given file path and return it as a NumPy array.

    Parameters
    ----------
    path : str
        Path to the video file.

    Returns
    -------
    np.ndarray or None
        A NumPy array of shape (num_frames, H, W, C) if successful, otherwise
        None if an error occurs.

    """
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for _i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(
    rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None
):
    """Draw a circle on the given PIL image at the specified coordinate.

    Parameters
    ----------
    rgb : PIL.Image.Image
        The image on which to draw.
    coord : tuple of float
        The (x, y) coordinate for the circle center.
    radius : int
        Radius of the circle.
    color : tuple of int, optional
        RGB color, by default (255, 0, 0).
    visible : bool, optional
        Whether the circle is visible, by default True.
    color_alpha : int, optional
        Alpha value for the circle color, by default None.

    Returns
    -------
    PIL.Image.Image
        The image with the circle drawn.

    """
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(
        list(color) + [color_alpha if color_alpha is not None else 255]
    )

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    """Draw a line between two points on a PIL image.

    Parameters
    ----------
    rgb : PIL.Image.Image
        The image on which to draw.
    coord_y : tuple of int
        The (x, y) coordinate of the first point.
    coord_x : tuple of int
        The (x, y) coordinate of the second point.
    color : tuple of int
        RGB color for the line.
    linewidth : int
        Thickness of the line.

    Returns
    -------
    PIL.Image.Image
        The image with the line drawn.

    """
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    """Blend two images with the given alpha, beta, and gamma parameters.

    Parameters
    ----------
    rgb : np.ndarray
        The first image as a NumPy array.
    alpha : float
        Weight for the first image.
    original : np.ndarray
        The second image as a NumPy array.
    beta : float
        Weight for the second image.
    gamma : float
        Scalar added to each sum.

    Returns
    -------
    np.ndarray
        The blended image as a NumPy array (dtype 'uint8').

    """
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Visualizer:
    """A class for visualizing tracked points on video frames.

    This class can overlay circles or lines representing tracked points onto
    video frames, and then save or display the result as a video or a tensor.
    """

    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        """Initialize the Visualizer with user-defined settings.

        Parameters
        ----------
        save_dir : str, optional
            Directory in which to save output videos, by default "./results".
        grayscale : bool, optional
            Whether to convert video frames to grayscale, by default False.
        pad_value : int, optional
            Padding in pixels around each frame, by default 0.
        fps : int, optional
            Frames per second for output video, by default 10.
        mode : str, optional
            Color mode for track visualization ('rainbow', 'cool', etc.),
            by default "rainbow".
        linewidth : int, optional
            Line/circle thickness, by default 2.
        show_first_frame : int, optional
            How many times to repeat the first frame, by default 10.
        tracks_leave_trace : int, optional
            Number of frames the track lines persist (-1 for infinite),
            by default 0.

        """
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,
        step: int = 0,
        query_frame=0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        opacity: float = 1.0,
    ):
        """Overlay tracks on a video and optionally save it.

        Parameters
        ----------
        video : torch.Tensor
            A tensor of shape (B, T, C, H, W).
        tracks : torch.Tensor
            A tensor of shape (B, T, N, 2) with tracked points.
        visibility : torch.Tensor, optional
            A boolean tensor of shape (B, T, N, 1) indicating visibility,
            by default None.
        gt_tracks : torch.Tensor, optional
            Ground-truth tracks of shape (B, T, N, 2), by default None.
        segm_mask : torch.Tensor, optional
            Segmentation mask of shape (B, 1, H, W), by default None.
        filename : str, optional
            Base name for the output video file, by default "video".
        writer : torch.utils.tensorboard.SummaryWriter, optional
            A SummaryWriter for TensorBoard, by default None.
        step : int, optional
            Global step for logging, by default 0.
        query_frame : int, optional
            Frame index used as a reference for some color modes,
            by default 0.
        save_video : bool, optional
            Whether to save the output as a video, by default True.
        compensate_for_camera_motion : bool, optional
            Whether to apply camera motion compensation, by default False.
        opacity : float, optional
            Alpha value for the circle color in the range
                [0, 1], by default 1.0.

        Returns
        -------
        torch.Tensor
            The final video with overlaid tracks, shape (1, T, C, H, W).

        """
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][
                coords[:, 1], coords[:, 0]
            ].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            color_alpha=color_alpha,
        )
        if save_video:
            self.save_video(
                res_video, filename=filename, writer=writer, step=step
            )
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        """Save a tensor of frames as a video file or to a TensorBoard writer.

        Parameters
        ----------
        video : torch.Tensor
            Video of shape (1, T, C, H, W).
        filename : str
            The base filename for saving the video.
        writer : torch.utils.tensorboard.SummaryWriter, optional
            If provided, the video is logged to TensorBoard, by default None.
        step : int, optional
            Global step for logging, by default 0.

        """
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [
                wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list
            ]

            # Prepare the video file path
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list[2:-1]:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def compute_vector_colors(self, tracks, T, N, query_frame, segm_mask):
        """Compute vector colors for each track over all frames.

        Parameters
        ----------
        tracks : np.ndarray
            Array of shape (T, N, 2) with track coordinates.
        T : int
            Total number of frames.
        N : int
            Number of tracks.
        query_frame : int
            Reference frame index.
        segm_mask : np.ndarray or None
            Segmentation mask.

        Returns
        -------
        np.ndarray
            Array of shape (T, N, 3) with RGB colors.

        """
        vector_colors = np.zeros((T, N, 3))
        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(
                tracks - tracks[query_frame][None]
            )
        elif segm_mask is None:
            if self.mode == "rainbow":
                y_min = tracks[query_frame, :, 1].min()
                y_max = tracks[query_frame, :, 1].max()
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    # Use query_frame directly if it is not a tensor.
                    qf = (
                        query_frame
                        if not isinstance(query_frame, torch.Tensor)
                        else query_frame[n]
                    )
                    color = self.color_map(norm(tracks[qf, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # Color changes with time.
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255
                valid = segm_mask > 0
                if np.any(valid):
                    y_min = tracks[0, valid, 1].min()
                    y_max = tracks[0, valid, 1].max()
                    norm = plt.Normalize(y_min, y_max)
                    for n in range(N):
                        if segm_mask[n] > 0:
                            color = self.color_map(norm(tracks[0, n, 1]))
                            color = np.array(color[:3])[None] * 255
                            vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                segm_mask = (
                    segm_mask.cpu() if hasattr(segm_mask, "cpu") else segm_mask
                )
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = (
                    np.array(self.color_map(1.0)[:3]) * 255.0
                )
                color[segm_mask <= 0] = (
                    np.array(self.color_map(0.0)[:3]) * 255.0
                )
                vector_colors = np.repeat(color[None], T, axis=0)
        return vector_colors

    def draw_tracks_and_points(
        self,
        res_video,
        tracks,
        vector_colors,
        visibility,
        query_frame,
        compensate_for_camera_motion,
        segm_mask,
        gt_tracks,
        color_alpha,
    ):
        """Draw track lines and points on each video frame.

        Parameters
        ----------
        res_video : list of np.ndarray
            List of video frames.
        tracks : np.ndarray
            Array of shape (T, N, 2) with track coordinates.
        vector_colors : np.ndarray
            Array of shape (T, N, 3) with RGB colors.
        visibility : np.ndarray or None
            Visibility array of shape (B, T, N, 1).
        query_frame : int
            Reference frame index.
        compensate_for_camera_motion : bool
            Flag indicating whether to compensate for camera motion.
        segm_mask : np.ndarray or None
            Segmentation mask.
        gt_tracks : np.ndarray or None
            Ground-truth tracks.
        color_alpha : int
            Alpha value for drawing.

        Returns
        -------
        list of np.ndarray
            Updated list of video frames with drawn tracks.

        """
        T, N, _ = tracks.shape

        # Draw track lines if tracks_leave_trace is nonzero.
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]
                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]
                res_video[t] = self._draw_pred_tracks(
                    res_video[t], curr_tracks, curr_colors
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        # Draw points on each frame.
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                # Set visibility using a single condition.
                visible = True if visibility is None else visibility[0, t, i]
                if (
                    coord[0] != 0
                    and coord[1] != 0
                    and (
                        not compensate_for_camera_motion
                        or (compensate_for_camera_motion and segm_mask[i] > 0)
                    )
                ):
                    img = draw_circle(
                        img,
                        coord=coord,
                        radius=int(self.linewidth * 2),
                        color=vector_colors[t, i].astype(int),
                        visible=visible,
                        color_alpha=color_alpha,
                    )
            res_video[t] = np.array(img)
        return res_video

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
    ):
        """Draw the given tracks on the video frames.

        Parameters
        ----------
        video : torch.Tensor
            A tensor of shape (B, T, C, H, W).
        tracks : torch.Tensor
            A tensor of shape (B, T, N, 2).
        visibility : torch.Tensor, optional
            Visibility tensor of shape (B, T, N, 1), by default None.
        segm_mask : torch.Tensor, optional
            Segmentation mask of shape (B, 1, H, W), by default None.
        gt_tracks : torch.Tensor, optional
            Ground-truth tracks, shape (B, T, N, 2), by default None.
        query_frame : int, optional
            The reference frame index, by default 0.
        compensate_for_camera_motion : bool, optional
            Whether to adjust for camera motion, by default False.
        color_alpha : int, optional
            Alpha value for the track colors in [0, 255], by default 255.

        Returns
        -------
        torch.Tensor
            The final video with tracks drawn, shape (1, T, C, H, W).

        """
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3

        # Convert video and tracks to NumPy arrays.
        video_np = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()
        tracks_np = tracks[0].long().detach().cpu().numpy()
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        # Copy video frames.
        res_video = [frame.copy() for frame in video_np]

        # Compute vector colors using helper method.
        vector_colors = self.compute_vector_colors(
            tracks_np, T, N, query_frame, segm_mask
        )

        # Draw tracks and points.
        res_video = self.draw_tracks_and_points(
            res_video,
            tracks_np,
            vector_colors,
            visibility,
            query_frame,
            compensate_for_camera_motion,
            segm_mask,
            gt_tracks,
            color_alpha,
        )

        # Optionally, repeat the first frame.
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        final_video = torch.from_numpy(np.stack(res_video))
        return final_video.permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        """Draw predicted tracks (lines) on a single video frame.

        Parameters
        ----------
        rgb : np.ndarray
            The current frame as a NumPy array of shape (H, W, 3).
        tracks : np.ndarray
            The track coordinates of shape (T, N, 2).
        vector_colors : np.ndarray
            The colors of shape (T, N, 3).
        alpha : float, optional
            Weight for blending lines, by default 0.5.

        Returns
        -------
        np.ndarray
            The updated frame with predicted tracks drawn.

        """
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self.linewidth,
                    )
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb),
                            alpha,
                            np.array(original),
                            1 - alpha,
                            0,
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        """Draw ground-truth tracks (as red crosses) on a single video frame.

        Parameters
        ----------
        rgb : np.ndarray
            The current frame of shape (H, W, 3).
        gt_tracks : np.ndarray
            Ground-truth coordinates of shape (T, N, 2).

        Returns
        -------
        np.ndarray
            The updated frame with ground-truth tracks drawn.

        """
        T, N, _ = gt_tracks.shape
        color = np.array((211, 0, 0))
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (
                        int(gt_tracks[0]) + length,
                        int(gt_tracks[1]) + length,
                    )
                    coord_x = (
                        int(gt_tracks[0]) - length,
                        int(gt_tracks[1]) - length,
                    )
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
                    coord_y = (
                        int(gt_tracks[0]) - length,
                        int(gt_tracks[1]) + length,
                    )
                    coord_x = (
                        int(gt_tracks[0]) + length,
                        int(gt_tracks[1]) - length,
                    )
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
        rgb = np.array(rgb)
        return rgb
