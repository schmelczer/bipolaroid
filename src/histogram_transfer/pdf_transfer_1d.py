import numpy as np


def pdf_transfer_1d(pX: np.ndarray, pY: np.ndarray) -> np.ndarray:
    PX = np.cumsum(pX + np.finfo(float).eps)
    PX /= PX[-1]

    PY = np.cumsum(pY + np.finfo(float).eps)
    PY /= PY[-1]

    f = np.interp(PX, PY, np.arange(len(pX)))

    return f
