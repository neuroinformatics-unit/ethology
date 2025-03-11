import torch
import imageio.v3 as iio
import numpy as np
from ethology.visualisation.visualizer import Visualizer

def load_video_frames(video_path_or_url):
    """
    Load video frames using imageio.
    Returns:
        frames (np.ndarray): Array of shape (num_frames, H, W, C).
    """
    frames = iio.imread(video_path_or_url, plugin="FFMPEG")
    frames = np.array(frames)
    return frames

def preprocess_video(frames, device="cuda"):
    """
    Convert frames to a torch.Tensor of shape (B, T, C, H, W) on the specified device.
    """
    # Convert frames to a tensor, permute dimensions to (T, C, H, W) and add batch dimension.
    video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    return video_tensor

def convert_query_points(input_coords, device="cuda"):
    """
    Convert input coordinates from [frame, y, x] to a tensor of shape (num_points, 2) in [x, y] order.
    
    Args:
        input_coords (np.ndarray): Array of shape (num_points, 3) in [frame, y, x] format.
        
    Returns:
        query_points (torch.Tensor): Tensor of shape (num_points, 2) in [x, y] order.
    """
    # Drop the frame column (assumes all points come from the same frame)
    # and reorder from [y, x] to [x, y]
    
    # Swap columns 1 and 2 to change [frame, y, x] to [frame, x, y]
    formatted = input_coords.copy()
    formatted[:, [1, 2]] = formatted[:, [2, 1]]
    
    # Add batch dimension: shape becomes (1, N, 3)
    queries = formatted[np.newaxis, ...]
    
    # Convert to torch.Tensor
    queries = torch.tensor(queries, dtype=torch.float32, device=device)
    return queries

def run_co_tracker(video_path_or_url, input_coords, device="cuda", weights_path=None):
    """
    Run the offline Co-Tracker model on the given video and query points.
    
    Args:
        video_path_or_url (str): Path or URL to the video file.
        input_coords (np.ndarray): Array of shape (num_points, 3) with [frame, y, x] coordinates.
        device (str): Device to run the model on ("cuda" or "cpu").
        weights_path (str, optional): Path to a checkpoint (if you wish to override the default).
        
    Returns:
        pred_tracks (np.ndarray): Predicted tracks, shape (B, T, N, 2).
        pred_visibility (np.ndarray): Predicted visibility, shape (B, T, N, 1).
    """
    # 1. Load and preprocess the video
    frames = load_video_frames(video_path_or_url)
    video = preprocess_video(frames, device=device)
    print(f"[Co-Tracker] Loaded video with shape: {frames.shape}")

    # 2. Convert the input coordinates to the proper format
    query_points = convert_query_points(input_coords, device=device)

    # 3. Load the Co-Tracker model via torch.hub.
    # This uses the offline version as in the official demo.
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()

    # 4. Run inference with the custom query points.
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=query_points)
    
    # Convert outputs to NumPy arrays.
    # pred_tracks = pred_tracks.cpu().numpy()
    # pred_visibility = pred_visibility.cpu().numpy()
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    res_video = vis.visualize(video, pred_tracks, pred_visibility)
    return res_video, pred_tracks, pred_visibility
