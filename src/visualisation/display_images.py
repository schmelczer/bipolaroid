import matplotlib.pyplot as plt
from typing import Dict
from PIL.Image import Image
from math import ceil


def display_images(
    images: Dict[str, Image], images_per_row: int = 3, img_size_inches: int = 2
) -> plt.Figure:
    row_count = ceil(len(images) / images_per_row)

    an_image = next(iter(images.values()))
    aspect_ratio = an_image.size[0] / an_image.size[1]

    unit_height = (
        img_size_inches
        if an_image.size[1] > an_image.size[0]
        else img_size_inches / aspect_ratio
    )
    unit_width = (
        img_size_inches
        if an_image.size[0] > an_image.size[1]
        else img_size_inches * aspect_ratio
    )

    fig, axes = plt.subplots(
        nrows=row_count,
        ncols=images_per_row,
        figsize=(unit_width * images_per_row, unit_height * row_count),
    )

    axes = axes.flatten()

    for i, (title, image) in enumerate(images.items()):
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(title)

    for i in range(len(images), len(axes)):
        axes[i].axis("off")

    fig.subplots_adjust(
        hspace=0.25, wspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95
    )

    return fig
