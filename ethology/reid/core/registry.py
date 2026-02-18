
from collections import OrderedDict
import torch
from ethology.reid.core.config import MODEL_TYPES  #, NR_CLASSES_DICT, TRAINED_URLS
from ethology.reid.core.factory import MODEL_FACTORY
# from ethology.utils import logger as LOGGER  # If needed, implement or set LOGGER

class ReIDModelRegistry:
	"""Encapsulates model registration and related utilities."""

	@staticmethod
	def show_downloadable_models():
		# LOGGER.info("Available .pt ReID models for automatic download")
		# LOGGER.info(list(TRAINED_URLS.keys()))
		pass

	@staticmethod
	def get_model_name(model):
		for name in MODEL_TYPES:
			if name in model.name:
				return name
		return None

	@staticmethod
	def get_model_url(model):
		# return TRAINED_URLS.get(model.name, None)
		return None

	@staticmethod
	def load_pretrained_weights(model, weight_path):
		device = "cpu" if not torch.cuda.is_available() else None
		checkpoint = torch.load(
			weight_path,
			map_location=torch.device("cpu") if device == "cpu" else None,
			weights_only=False,
			encoding='latin1',
		)
		state_dict = checkpoint.get("state_dict", checkpoint)
		model_dict = model.state_dict()
		new_state_dict = OrderedDict()
		matched_layers, discarded_layers = [], []
		for k, v in state_dict.items():
			key = k[7:] if k.startswith("module.") else k
			if key in model_dict and model_dict[key].size() == v.size():
				new_state_dict[key] = v
				matched_layers.append(key)
			else:
				discarded_layers.append(key)
		model_dict.update(new_state_dict)
		model.load_state_dict(model_dict)

	@staticmethod
	def show_available_models():
		# LOGGER.info("Available models:")
		# LOGGER.info(list(MODEL_FACTORY.keys()))
		pass

	@staticmethod
	def get_nr_classes(weights):
		# dataset_key = weights.name.split("_")[1]
		# return NR_CLASSES_DICT.get(dataset_key, 1)
		return 1

	@staticmethod
	def build_model(name, weights, num_classes, loss="softmax", pretrained=True, use_gpu=True):
		if name not in MODEL_FACTORY:
			available = list(MODEL_FACTORY.keys())
			raise KeyError(f"Unknown model '{name}'. Must be one of {available}")
		return MODEL_FACTORY[name](
			num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu
		)
