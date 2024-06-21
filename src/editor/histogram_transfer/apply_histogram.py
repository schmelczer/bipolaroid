from editor.histogram_transfer import pdf_transfer_3d
import numpy as np
from scipy.ndimage import zoom


def apply_histogram(original_image, target_histogram, bin_count: int):
    actual_predicted_histogram = target_histogram.cpu().detach().numpy().squeeze()

    scale = 64 / bin_count
    scaled_predicted_histogram = zoom(actual_predicted_histogram, scale, order=3)
    scaled_predicted_histogram = (
        scaled_predicted_histogram / scaled_predicted_histogram.sum()
    )

    [h, w, _] = np.array(original_image).shape

    histogram = np.round(scaled_predicted_histogram * h * w).astype(int)

    rgb_vectors = []

    for r in range(histogram.shape[0]):
        for g in range(histogram.shape[1]):
            for b in range(histogram.shape[2]):
                # Append the RGB value 'count' times to the list
                for _ in range(histogram[r, g, b]):
                    rgb_vectors.append([r, g, b])

    rgb_vectors = np.array(rgb_vectors)
    np.random.shuffle(rgb_vectors)
    rgb_vectors = rgb_vectors * 256 / 64

    return pdf_transfer_3d(
        source=np.array(original_image),
        target_flattened=rgb_vectors.transpose(),
        relaxation=0.9,
        bin_count=3500,
        iterations=50,
        smoothness=1,
        should_regrain=True,
    )
