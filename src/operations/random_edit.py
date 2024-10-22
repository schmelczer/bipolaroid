from PIL import Image, ImageEnhance
from utils import random
from operations import (
    add_noise,
    add_random_colour_spill,
    get_random_gamma,
    adjust_gamma,
    apply_pixel_shader,
    get_random_brightness_lut,
    get_random_saturation_per_hue_lut,
)
import numpy as np


def random_edit(img: Image, seed: int = 42) -> Image:
    np.random.seed(seed)

    ratio_of_unchanged_images = 0.95
    if np.random.uniform(0, 1) > ratio_of_unchanged_images:
        return img

    img = img.convert("RGB")

    img = ImageEnhance.Contrast(img).enhance(random(0.5, 1.5))
    img = adjust_gamma(img, get_random_gamma())

    img = img.convert("HSV")
    saturation_lut = get_random_saturation_per_hue_lut()
    brightness_lut = get_random_brightness_lut()
    img = apply_pixel_shader(
        img, lambda h, s, v: (h, round(s * saturation_lut[h]), brightness_lut[v])
    )
    img = img.convert("RGB")

    img = add_random_colour_spill(img, 0.2)
    img = add_noise(img, random(0, 0.1))

    return img
