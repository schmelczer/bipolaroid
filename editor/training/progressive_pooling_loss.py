from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgressivePoolingLoss(nn.Module):
    def __init__(self, target_sizes: List[int], damping: float):
        super(ProgressivePoolingLoss, self).__init__()
        self._target_sizes = target_sizes
        self._damping = damping

    def forward(self, tensor_a, tensor_b):
        assert (
            tensor_a.size() == tensor_b.size()
        ), f"Input tensors must have the same size, got {tensor_a.size()} and {tensor_b.size()}"

        assert (
            len(tensor_a.size()) == 5
        ), f"Input tensors must have 5 dimensions, got {tensor_a.size()}"

        _minibatch_size, _channels, depth, height, width = tensor_a.size()
        assert depth == height == width, "Input tensors must be cubes."

        loss = 0.0
        weight = 1

        for target_size in self._target_sizes:
            pool_size = depth // target_size
            pooled_a = F.avg_pool3d(tensor_a, pool_size) * (pool_size**3)
            pooled_b = F.avg_pool3d(tensor_b, pool_size) * (pool_size**3)

            diff = torch.abs(pooled_a - pooled_b)

            loss += diff.mean() * weight
            weight *= self._damping

        return loss
