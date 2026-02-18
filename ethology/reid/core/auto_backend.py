
from pathlib import Path
from typing import Tuple, Union
import torch
from ethology.reid.backends.onnx_backend import ONNXBackend
from ethology.reid.backends.openvino_backend import OpenVinoBackend
from ethology.reid.backends.pytorch_backend import PyTorchBackend
try:
	from ethology.reid.backends.tensorrt_backend import TensorRTBackend
except ImportError:
	class TensorRTBackend:
		def __init__(self, *args, **kwargs):
			raise ImportError("TensorRT and pycuda are required for TensorRTBackend. Please install them and ensure libcudnn.so.8 is available in LD_LIBRARY_PATH.")
from ethology.reid.backends.tflite_backend import TFLiteBackend
from ethology.reid.backends.torchscript_backend import TorchscriptBackend
# from ethology.reid.core import export_formats  # If needed, implement or copy export_formats
# from ethology.utils import WEIGHTS  # If needed, implement or set WEIGHTS
# from ethology.utils import logger as LOGGER  # If needed, implement or set LOGGER
# from ethology.utils.torch_utils import select_device  # If needed, implement or set select_device

class ReidAutoBackend:
	def __init__(
		self,
		weights: Path,
		device: torch.device = torch.device("cpu"),
		half: bool = False,
	):
		super().__init__()
		w = weights[0] if isinstance(weights, list) else weights
		(
			self.pt,
			self.pth,
			self.jit,
			self.onnx,
			self.xml,
			self.engine,
			self.tflite,
		) = self.model_type(w)
		self.weights = weights
		self.device = device  # For simplicity, skip select_device for now
		self.half = half
		self.model = self.get_backend()

	def get_backend(self):
		backend_map = {
			self.pt or self.pth: PyTorchBackend,
			self.jit: TorchscriptBackend,
			self.onnx: ONNXBackend,
			self.engine: TensorRTBackend,
			self.xml: OpenVinoBackend,
			self.tflite: TFLiteBackend,
		}
		for condition, backend_class in backend_map.items():
			if condition:
				return backend_class(self.weights, self.device, self.half)
		raise RuntimeError("This model framework is not supported yet!")

	def check_suffix(self, file: Path = "osnet_x0_25_msmt17.pt", suffix: Union[str, Tuple[str, ...]] = (".pt",), msg: str = ""):
		suffix = [suffix] if isinstance(suffix, str) else list(suffix)
		files = [file] if isinstance(file, (str, Path)) else list(file)
		for f in files:
			file_suffix = Path(f).suffix.lower()
			if file_suffix and file_suffix not in suffix:
				print(f"File {f} does not have an acceptable suffix. Expected: {suffix}")

	def model_type(self, p: Path) -> Tuple[bool, ...]:
		# For demo, just check for .pt
		sf = [".pt", ".pth", ".jit", ".onnx", ".xml", ".engine", ".tflite"]
		self.check_suffix(p, sf)
		types = [str(Path(p)).endswith(s) for s in sf]
		# OpenVINO explicit check
		if Path(p).suffix in ['.xml', '.bin']:
			types[3] = True
		return tuple(types)
