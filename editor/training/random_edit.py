from PIL import Image, ImageEnhance
from ..utils import random, get_colour_lut, apply_pixel_shader
from ..operations import add_noise, add_random_colour_spill
import numpy as np


def random_edit(img: Image, seed: int = 42) -> Image:
    np.random.seed(seed)
    img = add_noise(img, random(0, 0.2))
    img = ImageEnhance.Contrast(img).enhance(random(0.5, 2))
    img = add_random_colour_spill(img, 1.3)
    img = img.convert("HSV")
    saturation_lut = get_colour_lut(variance=0.3, count=5, type="linear")
    brightness_lut = get_colour_lut(variance=0.3, count=5, type="cubic")
    img = apply_pixel_shader(
        img, lambda h, s, v: (h, saturation_lut[s], brightness_lut[v])
    )
    img = img.convert("RGB")
    return img
