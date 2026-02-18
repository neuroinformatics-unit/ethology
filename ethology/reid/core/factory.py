
# Import model constructors from ethology's local backbones
from ethology.reid.backbones.hacnn import HACNN
from ethology.reid.backbones.mlfn import mlfn
from ethology.reid.backbones.mobilenetv2 import mobilenetv2_x1_0, mobilenetv2_x1_4
from ethology.reid.backbones.osnet import osnet_ibn_x1_0, osnet_x0_5, osnet_x0_25, osnet_x0_75, osnet_x1_0
from ethology.reid.backbones.osnet_ain import osnet_ain_x0_5, osnet_ain_x0_25, osnet_ain_x0_75, osnet_ain_x1_0
from ethology.reid.backbones.resnet import resnet50, resnet101
# from ethology.reid.backbones.lmbn.lmbn_n import LMBN_n  # If present
# from ethology.reid.backbones.clip.make_model import make_model  # If present

MODEL_FACTORY = {
	"resnet50": resnet50,
	"resnet101": resnet101,
	"mobilenetv2_x1_0": mobilenetv2_x1_0,
	"mobilenetv2_x1_4": mobilenetv2_x1_4,
	"hacnn": HACNN,
	"mlfn": mlfn,
	"osnet_x1_0": osnet_x1_0,
	"osnet_x0_75": osnet_x0_75,
	"osnet_x0_5": osnet_x0_5,
	"osnet_x0_25": osnet_x0_25,
	"osnet_ibn_x1_0": osnet_ibn_x1_0,
	"osnet_ain_x1_0": osnet_ain_x1_0,
	"osnet_ain_x0_75": osnet_ain_x0_75,
	"osnet_ain_x0_5": osnet_ain_x0_5,
	"osnet_ain_x0_25": osnet_ain_x0_25,
	# "lmbn_n": LMBN_n,  # Uncomment if implemented
	# "clip": make_model,  # Uncomment if implemented
}
