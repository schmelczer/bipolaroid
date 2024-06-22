import torch.nn as nn
import torch.nn.functional as F


class Dummy(nn.Module):
    def __init__(self, **_):
        super(Dummy, self).__init__()

    def forward(self, x):
        return x
