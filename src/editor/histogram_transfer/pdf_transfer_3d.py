import numpy as np
from editor.utils import generate_rotation_matrices
from editor.histogram_transfer import pdf_transfer_1d
from editor.histogram_transfer import regrain


EPSILON = 1e-6


def pdf_transfer_3d(
    source: np.ndarray,
    target_flattened: np.ndarray,
    relaxation: float = 1,
    bin_count: int = 1000,
    iterations: int = 25,
    smoothness: float = 1,
    should_regrain: bool = True,
):
    [h, w, c] = source.shape
    source_flattened = source.reshape(-1, c).transpose()

    rotation_matrices = generate_rotation_matrices(iterations)
    for i, rotation in enumerate(rotation_matrices, start=1):
        D0R = rotation @ source_flattened
        D1R = rotation @ target_flattened
        D0R_ = np.zeros_like(source_flattened)

        for i in range(rotation.shape[0]):
            datamin = min(np.min(D0R[i, :]), np.min(D1R[i, :])) - EPSILON
            datamax = max(np.max(D0R[i, :]), np.max(D1R[i, :])) + EPSILON
            u = np.linspace(datamin, datamax, bin_count)

            p0R, _ = np.histogram(D0R[i, :], bins=u, density=True)
            p1R, _ = np.histogram(D1R[i, :], bins=u, density=True)

            f = pdf_transfer_1d(p0R, p1R)
            mapped_values = (
                np.interp(D0R[i, :], u[:-1], f) * (datamax - datamin) / (bin_count - 1)
                + datamin
            )
            D0R_[i, :] = mapped_values

        source_flattened = source_flattened + relaxation * (rotation.T @ (D0R_ - D0R))
        source_flattened.clip(0, 255, out=source_flattened)

    result = source_flattened.astype(np.uint8).transpose().reshape(h, w, c)
    return regrain(source, result, smoothness=smoothness) if should_regrain else result
