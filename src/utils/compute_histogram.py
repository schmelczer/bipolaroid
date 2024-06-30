from PIL import Image
import numpy as np


def compute_histogram(
    image: Image.Image | np.ndarray,
    bins: int = 16,
    value_range=(0, 256),
    normalize: bool = True,
) -> np.ndarray:
    image = np.array(image) if isinstance(image, Image.Image) else image

    histogram, _ = np.histogramdd(
        image.reshape(-1, 3), bins=bins, range=[value_range, value_range, value_range]
    )

    histogram = histogram.astype(np.float32)

    if normalize:
        histogram = histogram / np.sum(histogram)

    return histogram
