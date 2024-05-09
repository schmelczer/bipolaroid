import numpy as np
from typing import List
from .random import random
from .interpolate import interpolate, INTERPOLATION_TYPE


def get_edit_points(variance: float, count: int) -> List[float]:
    return [
        random(i / (count - 1) - variance, i / (count - 1) + variance)
        for i in range(count)
    ]


def get_colour_lut(
    variance=0.1, count=5, type: INTERPOLATION_TYPE = "cubic"
) -> List[int]:
    edit_points = get_edit_points(variance=variance, count=count)
    return [
        round(interpolate(edit_points, i / 255, type=type) * 255)
        for i in np.linspace(0, 255, 256)
    ]
