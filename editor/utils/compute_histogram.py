from PIL import Image
import numpy as np


def compute_histogram(image_path, bins: int, value_range=(0, 256)):
    image = Image.open(image_path)
    image = np.array(image)

    histogram, _ = np.histogramdd(
        image.reshape(-1, 3), bins=bins, range=[value_range, value_range, value_range]
    )
    histogram = histogram / np.sum(histogram)

    return histogram.astype(np.float32)
