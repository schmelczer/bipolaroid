from .v1 import HistogramRestorationNet as v1
from .simple_cnn import SimpleCNN
from .residual import Residual
from .normalised_cnn import NormalisedCNN
from .smart_res import SmartRes
from .attention_net import AttentionNet
from .res2 import Res2
from .attention2 import EnhancedAestheticHistogramNet
from .attention import PhotoEnhanceNetAdvanced
from .advanced_attention import PhotoEnhanceNetAdvanced as advanced_attention
import torch


MODELS = {
    # "v1": v1,
    "SimpleCNN": SimpleCNN,
    "Residual": Residual,
    # "NormalisedCNN": NormalisedCNN,
    # "SmartRes": SmartRes,
    # "AttentionNet": AttentionNet,
    # "attention2": EnhancedAestheticHistogramNet,
    # "advanced_attention": advanced_attention,
    # "Res2": Res2,
    # "attention1": PhotoEnhanceNetAdvanced,
}


def create_model(type: str, bin_count: int):
    return MODELS[type](bin_count)


def test_models():
    for model_name, model_constructor in MODELS.items():
        print(f"Testing model {model_name}")
        _test_network_dimensions(model_constructor)


def _test_network_dimensions(constructor):
    for bin_count in [16, 32, 64]:
        model = constructor(bin_count=bin_count)

        # Create a dummy input tensor of the correct shape
        input_tensor = torch.rand(4, 1, bin_count, bin_count, bin_count)

        # Test the model output
        output = model(input_tensor)
        assert (
            input_tensor.shape == output.shape
        ), f"Expected output shape {input_tensor.shape}, but got {output.shape}"
    print("Test passed! Output shape matches input shape.")
