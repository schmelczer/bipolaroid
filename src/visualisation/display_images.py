import matplotlib.pyplot as plt
from typing import Dict
from PIL.Image import Image
from math import ceil


def display_images(images: Dict[str, Image], images_per_row: int = 3, figsize=(24, 16)):
    fig, axes = plt.subplots(
        nrows=ceil(len(images) / images_per_row),
        ncols=min(images_per_row, len(images)),
        figsize=figsize,
    )

    axes = axes.flatten()

    for i, (title, image) in enumerate(images.items()):
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(title)

    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
