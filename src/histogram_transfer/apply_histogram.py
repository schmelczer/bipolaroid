from typing import Optional
import numpy as np
from utils import generate_rotation_matrices, compute_histogram
from histogram_transfer.pdf_transfer_1d import pdf_transfer_1d
from histogram_transfer.regrain import regrain


def apply_histogram(
    source_img: np.ndarray,
    target_histogram: np.ndarray,
    *,
    iterations: int = 25,
    source_histogram: Optional[np.ndarray] = None,
    should_regrain: bool = True,
):
    if not isinstance(source_img, np.ndarray):
        source_img = np.array(source_img)

    assert (
        target_histogram.shape[0]
        == target_histogram.shape[1]
        == target_histogram.shape[2]
    ), "Histograms must be 3D"

    bins = target_histogram.shape[0]
    assert 256 % bins == 0, "Bin size must be a factor of 256"

    if source_histogram is None:
        source_histogram = compute_histogram(source_img, bins=bins)
    else:
        assert (
            source_histogram.shape == target_histogram.shape
        ), "Source and target histograms must be the same shape"

    source_colours = np.array(
        [
            index for index, value in np.ndenumerate(source_histogram) if value > 0
        ]  # we ignore colours without occurences
    ).T  # unique rgb colours, 3xN
    source_colours_original = source_colours.copy()
    source_counts = np.array(
        [value for value in np.nditer(source_histogram) if value > 0]
    )  # cardinality of each rgb colour in source_colours, N

    target_histogram_colours = np.array(
        [index for index, value in np.ndenumerate(target_histogram) if value > 0]
    ).T
    target_histogram_counts = np.array(
        [value for value in np.nditer(target_histogram) if value > 0]
    )

    for rotation in generate_rotation_matrices(iterations):
        source_colour_adjustment = np.zeros_like(source_colours)
        rotated_source_colours = rotation @ source_colours
        rotated_target_histogram_colours = rotation @ target_histogram_colours

        rotated_source_colours.round(out=rotated_source_colours)
        rotated_target_histogram_colours.round(out=rotated_target_histogram_colours)

        assert rotation.shape[0] == 3
        for i in range(rotation.shape[0]):  # for each axis (rgb)
            sorted_source_colours = sorted(
                enumerate(source_counts),
                key=lambda x: rotated_source_colours[i, x[0]],
            )
            sorted_source_counts = [v for _, v in sorted_source_colours]
            sorted_source_indices = [i for i, _ in sorted_source_colours]

            sorted_target_histogram_counts = [
                v
                for _, v in sorted(
                    enumerate(target_histogram_counts),
                    key=lambda x: rotated_target_histogram_colours[i, x[0]],
                )
            ]
            sorted_target_histogram_colours = [
                v for v in sorted(rotated_target_histogram_colours[i, :])
            ]

            sorted_new_source_colours = pdf_transfer_1d(
                sorted_source_counts,
                sorted_target_histogram_counts,
                sorted_target_histogram_colours,
            )
            source_colour_adjustment[i, :] = [
                v
                for _, v in sorted(
                    enumerate(sorted_new_source_colours),
                    key=lambda x: sorted_source_indices[x[0]],
                )
            ]

        source_colours = source_colours + (
            rotation.T @ (source_colour_adjustment - rotated_source_colours)
        )
        source_colours.clip(0, bins, out=source_colours)

    lut = np.zeros((256, 256, 256, 3), dtype=np.uint8)
    scale = 256 // bins
    source_colours = source_colours.T * scale
    source_colours.clip(0, 256, out=source_colours)
    source_colours.round(out=source_colours)
    for old, new in zip(source_colours_original.T, source_colours):
        for x in range(scale):
            for y in range(scale):
                for z in range(scale):
                    lut[old[0] * scale + x, old[1] * scale + y, old[2] * scale + z] = (
                        new
                    )

    result = lut[source_img[:, :, 0], source_img[:, :, 1], source_img[:, :, 2]]
    return regrain(source_img, result) if should_regrain else result
