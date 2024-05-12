from .v1 import HistogramRestorationNet as v1
from .simple_cnn import SimpleCNN
from .residual import Residual
from .normalised_cnn import NormalisedCNN
from .smart_res import SmartRes
from .attention_net import AttentionNet
from .res2 import Res2


def create_model(type: str, bin_count: int):
    return {
        # "v1": v1,
        "SimpleCNN": SimpleCNN,
        "Residual": Residual,
        "NormalisedCNN": NormalisedCNN,
        "SmartRes": SmartRes,
        "AttentionNet": AttentionNet,
        "Res2": Res2,
    }[type](bin_count)
