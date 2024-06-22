from typing import Any, Dict, Tuple
from .v1 import HistogramRestorationNet as v1
from .simple_cnn import SimpleCNN
from .residual import Residual
from .residual2 import Residual2
from .residual3 import Residual3
from .dummy import Dummy
import torch
import torch.nn as nn
from pathlib import Path
import logging
import json


MODELS = {
    # "v1": v1,
    "Dummy": Dummy,
    "SimpleCNN": SimpleCNN,
    "Residual": Residual,
    "Residual2": Residual2,
    "Residual3": Residual3,
}


def create_model(type: str, bin_count: int):
    return MODELS[type](bin_count)


def save_model(model: nn.Module, hyperparameters: Dict[str, Any], path: Path):
    model_path = path.with_suffix(".pth")
    params_path = path.with_suffix(".json")

    logging.info(f"Saving model to {model_path}")
    logging.info(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
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
    logging.info(f"Parameter count: {sum(p.numel() for p in model.parameters())}")

    return model, hyperparameters


def test_models():
    for model_name, model_constructor in MODELS.items():
        logging.info(f"Testing model {model_name}")
        _test_network_dimensions(model_constructor)


def _test_network_dimensions(constructor):
    for bin_count in [16, 32, 64]:
        model = constructor()

        # Create a dummy input tensor of the correct shape, the mini-batch size is 4
        input_tensor = torch.rand(4, 1, bin_count, bin_count, bin_count)

        output = model(input_tensor)
        assert (
            input_tensor.shape == output.shape
        ), f"Expected output shape {input_tensor.shape}, but got {output.shape}"
    logging.info("Test passed! Output shape matches input shape.")
