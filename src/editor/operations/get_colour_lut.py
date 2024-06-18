import numpy as np
from typing import List
from ..utils.random import random
from .interpolate import interpolate, INTERPOLATION_TYPE


def get_random_saturation_per_hue_lut(
    variance: float = 0.4, count: int = 12
) -> List[int]:
    edit_points = [random(-variance, variance) + 1 for _ in range(count)]

    return [
        interpolate(edit_points, i / 255, type="cubic")
        for i in np.linspace(0, 255, 256)
    ]


def get_random_brightness_lut(
    variance: float = 0.2,
    count: int = 6,
    type: INTERPOLATION_TYPE = "linear",
    min_spectrum_size: float = 0.6,
    max_spectrum_size: float = 1.15,
) -> List[int]:
    spectrum_size = np.random.uniform(min_spectrum_size, max_spectrum_size)
    spectrum_start = np.random.uniform(0, max(0, 1 - spectrum_size))
    edit_points = sorted(
        [
            spectrum_start,
            *[
                spectrum_start
                + i * spectrum_size / (count - 2)
                + np.random.uniform(-variance, variance)
                for i in range(1, count - 2)
            ],
            spectrum_start + spectrum_size,
        ]
    )

    return [
        round(interpolate(edit_points, i / 255, type=type) * 255)
        for i in np.linspace(0, 255, 256)
    ]
