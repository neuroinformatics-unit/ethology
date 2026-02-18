"""
Example: Using the new ReID trajectory utility with MOT results

This script demonstrates how to use the ethology reid trajectory handler with a sample MOT output.
"""

from ethology.reid.core.reid_handler import ReIDTrajectoryHandler

# Example: Dummy MOT results (replace with your actual MOT output)
mot_results = [
    {'id': 1, 'trajectory': [(0, 0), (1, 1), (2, 2)]},
    {'id': 2, 'trajectory': [(5, 5), (6, 6), (7, 7)]},
]

# Initialize the handler (adjust parameters as needed)
reid_handler = ReIDTrajectoryHandler(model_name='osnet', device='cpu')

# Run re-identification on the MOT results
reid_results = reid_handler.reidentify(mot_results)

print('ReID Results:')
for item in reid_results:
    print(item)
