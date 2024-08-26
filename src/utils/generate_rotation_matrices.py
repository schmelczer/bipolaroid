from random import shuffle
from typing import List, Tuple
import numpy as np
from functools import lru_cache
from numpy.typing import NDArray


@lru_cache
def generate_rotation_matrices(count: int) -> List[NDArray[np.float64]]:
    axes = fibonacci_sphere(count)
    shuffle(axes)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    matrices = [_rotation_matrix(axis, angle) for axis, angle in zip(axes, angles)]
    for matrix in matrices:
        _check_rotation_matrix(matrix)
    return matrices


def fibonacci_sphere(samples: int) -> List[Tuple[float, float, float]]:
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    return points


def _rotation_matrix(
    axis: Tuple[float, float, float], theta: float
) -> NDArray[np.float64]:
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def _check_rotation_matrix(R: NDArray[np.float64]):
    # Check if the matrix is square
    assert R.shape == (3, 3), "Matrix must be 3x3"

    # Check orthogonality: R.T * R should be close to the identity matrix
    I = np.eye(3)
    assert np.allclose(np.dot(R.T, R), I)

    assert np.isclose(np.linalg.det(R), 1.0), f"det {np.linalg.det(R)}"
