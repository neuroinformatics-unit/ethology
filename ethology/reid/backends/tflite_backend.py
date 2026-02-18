from pathlib import Path

import numpy as np
import torch

from ethology.reid.backends.base_backend import BaseModelBackend
# Note: LOGGER can be replaced with print or a local logger if needed

class TFLiteBackend(BaseModelBackend):
	"""
	A class to handle TensorFlow Lite model inference with dynamic batch size support.
	"""
	def __init__(self, weights: Path, device: str, half: bool):
		super().__init__(weights, device, half)
		self.nhwc = True
		self.half = False

	def load_model(self, w):
		# self.checker.check_packages(("tensorflow",))
		print(f"Loading {str(w)} for TensorFlow Lite inference...")
		import tensorflow as tf
		self.interpreter = tf.lite.Interpreter(model_path=str(w))
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.current_allocated_batch_size = self.input_details[0]["shape"][0]

	def forward(self, im_batch: torch.Tensor) -> np.ndarray:
		im_batch = im_batch.cpu().numpy()
		batch_size = im_batch.shape[0]
		if batch_size != self.current_allocated_batch_size:
			self.interpreter.resize_tensor_input(
				self.input_details[0]["index"], [batch_size, 256, 128, 3]
			)
			self.interpreter.allocate_tensors()
			self.current_allocated_batch_size = batch_size
		self.interpreter.set_tensor(self.input_details[0]["index"], im_batch)
		self.interpreter.invoke()
		features = self.interpreter.get_tensor(self.output_details[0]["index"])
		return features
