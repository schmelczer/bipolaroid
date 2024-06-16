import numpy as np
from PIL import Image


def add_noise(img: Image, alpha: float) -> Image:
    width, height = img.size
    random_colors = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    random_img = Image.fromarray(random_colors)
    result = Image.blend(img, random_img, alpha)
    return result
