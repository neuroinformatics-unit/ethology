"""Test script to generate, load, and visualize keypoints data.

This script:
1. Generates a dummy SLEAP file with synthetic keypoint data
2. Loads it using ethology's load_keypoints
3. Visualizes the resulting 4D hypercube
"""

import numpy as np
import xarray as xr
from pathlib import Path
import tempfile

print("=" * 70)
print("KEYPOINTS FUNCTIONALITY TEST: Generate -> Load -> Visualize")
print("=" * 70)

# Step 1: Generate dummy SLEAP file
print("\n[STEP 1] Generating dummy SLEAP file...")
print("-" * 70)

try:
    import sleap_io as sio
    print("[OK] sleap-io imported")
except ImportError:
    print("[FAIL] sleap-io not installed. Install with: pip install sleap-io")
    exit(1)

# Create synthetic data parameters
n_frames = 10
n_instances = 2
n_keypoints = 5
keypoint_names = ["head", "tail", "left_ear", "right_ear", "nose"]

# Create skeleton
nodes = [sio.Node(name=name) for name in keypoint_names]
skeleton = sio.Skeleton(nodes=nodes, edges=[])

# Create a dummy video
video_path = "dummy_video.mp4"
video = sio.Video.from_filename(video_path)

# Generate labeled frames with keypoints
labeled_frames = []
np.random.seed(42)  # For reproducibility

for frame_idx in range(n_frames):
    # Create instances for this frame
    instances = []
    
    for inst_id in range(n_instances):
        # Generate keypoint coordinates with some variation
        # Instance 0: starts at (100, 100), moves right
        # Instance 1: starts at (200, 200), moves left
        base_x = 100 + inst_id * 100 + frame_idx * 5 * (1 if inst_id == 0 else -1)
        base_y = 100 + inst_id * 100 + np.sin(frame_idx * 0.5) * 20
        
        # Create points array: shape (n_keypoints, 2) for (x, y)
        points_array = np.full((n_keypoints, 2), np.nan, dtype=np.float64)
        
        for kp_idx, kp_name in enumerate(keypoint_names):
            # Add some offset for each keypoint
            offset_x = (kp_idx - 2) * 10  # Spread horizontally
            offset_y = (kp_idx % 2) * 5    # Small vertical variation
            
            # Make some keypoints missing (NaN) occasionally
            if frame_idx == 0 and kp_idx == 2 and inst_id == 0:
                # Missing keypoint in first frame - leave as NaN
                continue
            elif frame_idx == 5 and kp_idx == 0 and inst_id == 1:
                # Missing keypoint in middle frame - leave as NaN
                continue
            else:
                x = base_x + offset_x + np.random.normal(0, 2)
                y = base_y + offset_y + np.random.normal(0, 2)
                points_array[kp_idx, 0] = x
                points_array[kp_idx, 1] = y
        
        instance = sio.Instance(points=points_array, skeleton=skeleton)
        instances.append(instance)
    
    labeled_frame = sio.LabeledFrame(video=video, frame_idx=frame_idx)
    labeled_frame.instances = instances
    labeled_frames.append(labeled_frame)

# Create Labels object
labels = sio.Labels(labeled_frames=labeled_frames, skeletons=[skeleton])

# Save to temporary file
temp_dir = Path(tempfile.gettempdir())
sleap_file = temp_dir / "test_keypoints.slp"
sio.save_file(labels, sleap_file)

print(f"[OK] Generated SLEAP file: {sleap_file}")
print(f"  Frames: {n_frames}")
print(f"  Instances per frame: {n_instances}")
print(f"  Keypoints: {keypoint_names}")

# Step 2: Load using ethology
print("\n[STEP 2] Loading with ethology...")
print("-" * 70)

try:
    from ethology.io.annotations import load_keypoints
    
    ds = load_keypoints.from_files(sleap_file, format="SLEAP")
    print("[OK] Dataset loaded successfully")
    print(f"  Dataset shape: {ds.position.shape}")
    print(f"  Dimensions: {dict(ds.position.sizes)}")
except Exception as e:
    print(f"[FAIL] Loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Visualize the 4D hypercube
print("\n[STEP 3] Visualizing 4D hypercube...")
print("-" * 70)

print("\nDataset Structure:")
print(f"  Dimensions: {list(ds.sizes.keys())}")
print(f"  Data variables: {list(ds.data_vars.keys())}")
print(f"  Coordinates:")
for coord_name, coord_values in ds.coords.items():
    if len(coord_values) <= 10:
        print(f"    {coord_name}: {list(coord_values.values)}")
    else:
        print(f"    {coord_name}: {len(coord_values)} values (first 5: {list(coord_values.values[:5])})")

print("\nPosition Array Statistics:")
pos = ds.position.values
print(f"  Shape: {pos.shape}")
print(f"  Total elements: {pos.size}")
print(f"  Valid (non-NaN) keypoints: {np.isfinite(pos).sum()}")
print(f"  Missing (NaN) keypoints: {np.isnan(pos).sum()}")
print(f"  Missing percentage: {100 * np.isnan(pos).sum() / pos.size:.2f}%")

print("\nPer-Dimension Statistics:")
print(f"  image_id dimension: {ds.sizes['image_id']} frames")
print(f"  space dimension: {list(ds.space.values)} (x, y coordinates)")
print(f"  keypoint dimension: {ds.sizes['keypoint']} keypoints")
print(f"    Names: {list(ds.keypoint.values)}")
print(f"  id dimension: {ds.sizes['id']} instances per frame")

print("\nMissing Keypoints Analysis:")
for image_id in range(min(3, ds.sizes['image_id'])):
    for inst_id in range(ds.sizes['id']):
        frame_pos = ds.position.sel(image_id=image_id, id=inst_id)
        missing = np.isnan(frame_pos.values).sum()
        total = frame_pos.size
        if missing > 0:
            print(f"  Frame {image_id}, Instance {inst_id}: {missing}/{total} missing keypoints")

print("\nSample Data (First Frame, First Instance):")
sample = ds.position.sel(image_id=0, id=0)
print(f"  Shape: {sample.shape}")
print(f"  Keypoint coordinates:")
for kp_idx, kp_name in enumerate(ds.keypoint.values):
    x = sample.sel(space='x', keypoint=kp_name).values
    y = sample.sel(space='y', keypoint=kp_name).values
    if np.isnan(x) or np.isnan(y):
        print(f"    {kp_name}: MISSING (NaN)")
    else:
        print(f"    {kp_name}: ({x:.2f}, {y:.2f})")

print("\nConfidence Array (if present):")
if "confidence" in ds.data_vars:
    conf = ds.confidence.values
    print(f"  Shape: {conf.shape}")
    print(f"  Valid values: {np.isfinite(conf).sum()}")
    print(f"  Mean confidence: {np.nanmean(conf):.3f}")
    print(f"  Min confidence: {np.nanmin(conf):.3f}")
    print(f"  Max confidence: {np.nanmax(conf):.3f}")
else:
    print("  Not present in dataset")

print("\nVisibility Array (if present):")
if "visibility" in ds.data_vars:
    vis = ds.visibility.values
    print(f"  Shape: {vis.shape}")
    print(f"  Valid values: {np.isfinite(vis).sum()}")
    visible_count = (vis == 1.0).sum()
    print(f"  Visible keypoints: {visible_count}")
else:
    print("  Not present in dataset")

print("\nDataset Attributes:")
for key, value in ds.attrs.items():
    if isinstance(value, dict) and len(value) <= 5:
        print(f"  {key}: {value}")
    elif isinstance(value, dict):
        print(f"  {key}: dict with {len(value)} entries")
    else:
        print(f"  {key}: {value}")

# Step 3.5: Simple visualization plot
print("\n[STEP 3.5] Creating visualization plot...")
print("-" * 70)

try:
    import matplotlib.pyplot as plt
    
    # Plot keypoints for first 3 frames, both instances
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for frame_idx in range(min(3, ds.sizes['image_id'])):
        ax = axes[frame_idx]
        
        # Plot both instances
        for inst_id in range(ds.sizes['id']):
            frame_pos = ds.position.sel(image_id=frame_idx, id=inst_id)
            x_coords = frame_pos.sel(space='x').values
            y_coords = frame_pos.sel(space='y').values
            
            # Filter out NaN values
            valid_mask = np.isfinite(x_coords) & np.isfinite(y_coords)
            if valid_mask.sum() > 0:
                ax.scatter(
                    x_coords[valid_mask],
                    y_coords[valid_mask],
                    label=f'Instance {inst_id}',
                    s=50,
                    alpha=0.7,
                )
                
                # Annotate keypoint names
                for kp_idx, kp_name in enumerate(ds.keypoint.values):
                    if valid_mask[kp_idx]:
                        ax.annotate(
                            kp_name,
                            (x_coords[kp_idx], y_coords[kp_idx]),
                            fontsize=8,
                            alpha=0.6,
                        )
        
        ax.set_title(f'Frame {frame_idx}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Image coordinates
    
    plt.tight_layout()
    plot_file = temp_dir / "test_keypoints_plot.png"
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualization plot saved: {plot_file}")
    
except ImportError:
    print("[WARN] matplotlib not available, skipping plot")
except Exception as e:
    print(f"[WARN] Plot creation failed: {e}")
    print("  (This is optional, continuing...)")

# Step 4: Verify round-trip (optional)
print("\n[STEP 4] Testing round-trip (save and reload)...")
print("-" * 70)

try:
    from ethology.io.annotations import save_keypoints
    
    # Add required attrs for saving
    if "map_image_id_to_video" not in ds.attrs:
        ds.attrs["map_image_id_to_video"] = {
            i: str(video_path) for i in range(ds.sizes['image_id'])
        }
    
    roundtrip_file = temp_dir / "test_keypoints_roundtrip.slp"
    save_keypoints.to_file(ds, roundtrip_file, format="SLEAP")
    print(f"[OK] Saved to: {roundtrip_file}")
    
    # Reload
    ds2 = load_keypoints.from_files(roundtrip_file, format="SLEAP")
    print("[OK] Reloaded successfully")
    
    # Compare shapes
    if ds.position.shape == ds2.position.shape:
        print("[OK] Shapes match")
    else:
        print(f"[WARN] Shape mismatch: {ds.position.shape} vs {ds2.position.shape}")
    
    # Compare keypoint names
    if list(ds.keypoint.values) == list(ds2.keypoint.values):
        print("[OK] Keypoint names match")
    else:
        print(f"[WARN] Keypoint names differ")
    
except Exception as e:
    print(f"[WARN] Round-trip test failed: {e}")
    print("  (This is optional, continuing...)")

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
print(f"\nGenerated files:")
print(f"  Original: {sleap_file}")
if 'roundtrip_file' in locals():
    print(f"  Round-trip: {roundtrip_file}")
print(f"\nTo clean up, delete these files manually.")
