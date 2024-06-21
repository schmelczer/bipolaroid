from typing import Any, Dict, Tuple
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
import torch.nn as nn
from pathlib import Path
import logging
import json


MODELS = {
    # "v1": v1,
    # "SimpleCNN": SimpleCNN,
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


def save_model(model: nn.Module, hyperparameters: Dict[str, Any], path: Path):
    model_path = path.with_suffix(".pth")
    params_path = path.with_suffix(".json")

    logging.info(f"Saving model to {model_path}")
    with open(model_path, "wb") as f:
        torch.save(model.state_dict(), f)
    with open(params_path, "w") as f:
        json.dump(hyperparameters, f, indent=2)


def load_model(path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    logging.info(f"Loading model from {path}")

    params_path = path.with_suffix(".json")
    with open(params_path) as f:
        hyperparameters = json.load(f)
    logging.info(f"Hyperparameters: {hyperparameters}")

    model_path = path.with_suffix(".pth")
    model = create_model(
        type=hyperparameters["model_type"],
        bin_count=hyperparameters["bin_count"],
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model, hyperparameters


def _test_models():
    for model_name, model_constructor in MODELS.items():
        logging.info(f"Testing model {model_name}")
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


if __name__ == "__main__":
    _test_models()
