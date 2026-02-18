
import os
from abc import abstractmethod
from pathlib import Path
import cv2
import gdown
import numpy as np
import torch
from filelock import SoftFileLock
from ethology.reid.core.registry import ReIDModelRegistry
# from ethology.utils import logger as LOGGER  # If needed, implement or set LOGGER
# from ethology.utils.checks import RequirementsChecker  # If needed, implement or set RequirementsChecker

class BaseModelBackend:

    def __init__(self, weights, device, half):
        self.weights = weights[0] if isinstance(weights, list) else weights
        if isinstance(self.weights, str):
            self.weights = Path(self.weights)
        # LOGGER.info(self.weights)
        self.device = device
        self.half = half
        self.model = None
        # Support both string and torch.device for device
        if hasattr(self.device, 'type'):
            self.cuda = torch.cuda.is_available() and self.device.type != "cpu"
        else:
            self.cuda = torch.cuda.is_available() and self.device != "cpu"

        self.download_model(self.weights)
        self.model_name = ReIDModelRegistry.get_model_name(self.weights)

        self.model = ReIDModelRegistry.build_model(
            self.model_name,
            self.weights,
            num_classes=ReIDModelRegistry.get_nr_classes(self.weights),
            pretrained=not (self.weights and self.weights.is_file()),
            use_gpu=device,
        )
        # self.checker = RequirementsChecker()

        self.load_model(self.weights)

        self.mean_array = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std_array = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if "clip" in self.model_name:
            self.mean_array = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
            self.std_array = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

        if "vehicleid" in self.weights.name or "veri" in self.weights.name:
            input_shape = (256, 256)
        elif "lmbn" in self.model_name:
            input_shape = (384, 128)
        elif "hacnn" in self.model_name:
            input_shape = (160, 64)
        else:
            input_shape = (256, 128)
        self.input_shape = input_shape


    def get_crops(self, xyxys, img):
        h, w = img.shape[:2]
        interpolation_method = cv2.INTER_LINEAR
        num_crops = len(xyxys)
        crops = torch.empty(
            (num_crops, 3, *self.input_shape),
            dtype=torch.half if self.half else torch.float,
            device=self.device,
        )
        for i, box in enumerate(xyxys):
            x1, y1, x2, y2 = box.round().astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(
                crop,
                (self.input_shape[1], self.input_shape[0]),
                interpolation=interpolation_method,
            )
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = torch.from_numpy(crop).to(
                self.device, dtype=torch.half if self.half else torch.float
            )
            crops[i] = torch.permute(crop, (2, 0, 1))
        crops = crops / 255.0
        crops = (crops - self.mean_array) / self.std_array
        return crops


    @torch.no_grad()
    def get_features(self, xyxys, img):
        if xyxys.size != 0:
            crops = self.get_crops(xyxys, img)
            crops = self.inference_preprocess(crops)
            features = self.forward(crops)
            features = self.inference_postprocess(features)
        else:
            features = np.array([])
        features = features / np.linalg.norm(features, axis=-1, keepdims=True)
        return features


    def warmup(self, imgsz=[(256, 128, 3)]):
        if self.device.type != "cpu":
            im = np.random.randint(0, 255, *imgsz, dtype=np.uint8)
            crops = self.get_crops(
                xyxys=np.array([[0, 0, 64, 64], [0, 0, 128, 128]]), img=im
            )
            crops = self.inference_preprocess(crops)
            self.forward(crops)


    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x


    def inference_preprocess(self, x):
        if self.half:
            if isinstance(x, torch.Tensor):
                if x.dtype != torch.float16:
                    x = x.half()
            elif isinstance(x, np.ndarray):
                if x.dtype != np.float16:
                    x = x.astype(np.float16)
        if hasattr(self, 'nhwc') and self.nhwc:
            if isinstance(x, torch.Tensor):
                x = x.permute(0, 2, 3, 1)
            elif isinstance(x, np.ndarray):
                x = np.transpose(x, (0, 2, 3, 1))
        return x


    def inference_postprocess(self, features):
        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0]) if len(features) == 1 else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)


    @abstractmethod
    def forward(self, im_batch):
        raise NotImplementedError("This method should be implemented by subclasses.")


    @abstractmethod
    def load_model(self, w):
        raise NotImplementedError("This method should be implemented by subclasses.")


    def download_model(self, w):
        if isinstance(w, str):
            w = Path(w)
        if w.suffix != ".pt":
            return
        model_url = ReIDModelRegistry.get_model_url(w)
        lock = SoftFileLock(str(w) + ".lock", timeout=300)
        with lock:
            if w.exists() or "openvino" in w.name:
                # LOGGER.info(f"[PID {os.getpid()}] Found existing ReID weights at {w}; skipping download.")
                return
            if model_url:
                # LOGGER.info(f"[PID {os.getpid()}] Downloading ReID weights from {model_url} → {w}")
                gdown.download(model_url, str(w), quiet=False)
            else:
                # LOGGER.error(
                #     f"No URL associated with the chosen ReID weights ({w}).\n"
                #     f"Choose one of the following:"
                # )
                ReIDModelRegistry.show_downloadable_models()
