import torch
import os


def get_device() -> torch.device:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
