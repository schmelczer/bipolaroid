import numpy as np


EPSILON = np.finfo(float).eps


def pdf_transfer_1d(
    source_y: np.ndarray, target_y: np.ndarray, target_x: np.ndarray = None
) -> np.ndarray:
    """
    return the best source_x-es
    """
    cumulative_source = np.cumsum(source_y).astype(np.float64)
    cumulative_source /= cumulative_source[-1]

    cumulative_target = np.cumsum(target_y).astype(np.float64)
    cumulative_target /= cumulative_target[-1]

    cumulative_transfered = np.interp(
        cumulative_source,
        cumulative_target,
        np.arange(len(cumulative_source)) if target_x is None else target_x,
    )
    return cumulative_transfered
