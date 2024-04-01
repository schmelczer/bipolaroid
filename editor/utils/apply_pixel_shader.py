from typing import Callable, Tuple
from PIL import Image


def apply_pixel_shader(
    img: Image, callback: Callable[[int, int, int], Tuple[int, int, int]]
):
    width, height = img.size
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            pixels[x, y] = callback(r, g, b)
    return img
