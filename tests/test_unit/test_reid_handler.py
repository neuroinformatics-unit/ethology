import numpy as np
from ethology.reid.core.handler import ReIDHandler

def test_extract_features_shape():
    handler = ReIDHandler(weights='osnet_x0_25_imagenet.pth')
    frame = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    dets = np.array([
        [10, 10, 50, 100, 0.9, 1],
        [60, 20, 100, 110, 0.8, 2],
    ])
    feats = handler.extract_features(frame, dets)
    assert feats.shape[0] == dets.shape[0]
