import numpy as np


def kldiv(P: np.ndarray, Q: np.ndarray) -> float:
    P /= P.sum()
    Q /= Q.sum()

    P_safe = np.maximum(P, np.finfo(float).eps)
    Q_safe = np.maximum(Q, np.finfo(float).eps)

    return np.sum(P_safe * np.log(P_safe / Q_safe))
