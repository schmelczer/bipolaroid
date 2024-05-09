from PIL import Image
from ..utils import random


def add_random_colour_spill(image: Image, range: float) -> Image:
    matrix = (
        random(1 / range, range),
        0.0,
        0.0,
        0.0,
        0.0,
        random(1 / range, range),
        0.0,
        0.0,
        0.0,
        0.0,
        random(1 / range, range),
        0.0,
    )
    return image.convert("RGB", matrix)
