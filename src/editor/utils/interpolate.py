import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Literal


INTERPOLATION_TYPE = Literal["cubic", "linear"]


def interpolate(
    control_points: List[float], t: float, type: INTERPOLATION_TYPE
) -> float:
    control_points = sorted(control_points)

    if type == "cubic":
        x = np.linspace(0, 1, len(control_points))
        cs = CubicSpline(x, control_points)
        return cs(t)

    if type == "linear":
        n = len(control_points) - 1
        segment_indices = np.linspace(0, 1, n + 1)

        index = np.searchsorted(segment_indices, t, side="right") - 1

        if t == 1:
            return control_points[-1]
        else:
            t_normalized = (t - segment_indices[index]) / (
                segment_indices[index + 1] - segment_indices[index]
            )
            return control_points[index] + t_normalized * (
                control_points[index + 1] - control_points[index]
            )

    raise ValueError("Invalid type")
