"""Detectors, pose estimation, detector+tracking models"""

# If we use Pytorch lightning, here we will have LightningModules for each model

# Pytorch lightning
# - all definitions of what will happen in training are in a LightningModule
# - The [lightning module](https://pytorch-lightning.readthedocs.io/en/1.5.10/common/lightning_module.html) holds all the core research ingredients:
# 		- The model
# 		- The optimizers
# 		- The train/ val/ test steps

# Maybe following ultralytics structure, with subdirectories model, predict, train, val
