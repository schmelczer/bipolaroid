import numpy as np
from random import choice
from PIL import Image


def adjust_gamma(img: Image, gamma: float) -> Image:
    return img.point(lambda x: (x / 255) ** gamma * 255)


def get_random_gamma() -> float:
    gamma = np.random.beta(1, 2) * 0.6
    return 1 / (1 + gamma) if choice([True, False]) else (gamma + 1)
