import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressivePoolingLoss(nn.Module):
    def __init__(self, initial_pool_size: int = 2, damping=1.8):
        super(ProgressivePoolingLoss, self).__init__()
        self._initial_pool_size = initial_pool_size
        self._damping = damping

    def forward(self, tensor_a, tensor_b):
        assert (
            tensor_a.size() == tensor_b.size()
        ), "Input tensors must have the same size."

        max_pool_size = min(tensor_a.size(1), tensor_a.size(2), tensor_a.size(3))

        loss = 0.0
        damping = 1

        for pool_size in range(self._initial_pool_size, max_pool_size):
            pooled_a = F.avg_pool3d(tensor_a, pool_size) * (pool_size**3)
            pooled_b = F.avg_pool3d(tensor_b, pool_size) * (pool_size**3)

            diff = torch.square(pooled_a - pooled_b)

            loss += diff.mean() / damping
            damping *= self._damping

        return loss
