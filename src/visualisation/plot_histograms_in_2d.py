import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_histograms_in_2d(
    histograms: Dict[str, np.ndarray], histograms_per_row=3, histograms_size_inches=4
):
    row_count = max(1, len(histograms) // histograms_per_row)
    fig = plt.figure(
        figsize=(
            histograms_per_row * histograms_size_inches,
            row_count * histograms_size_inches,
        )
    )

    for i, (title, histogram) in enumerate(histograms.items(), 1):
        ax = fig.add_subplot(row_count, histograms_per_row, i, projection="3d")

        size = histogram.shape[0]

        x, y, z = np.indices(histogram.shape)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = histogram.flatten()

        sizes = values * 5000  # this is just an arbitrary scaling factor

        colors = np.vstack((x, y, z)).T / (size - 1)

        sc = ax.scatter(x, y, z, c=colors, s=sizes, marker="o", alpha=0.5)

        ax.set_xlim([0, (size - 1)])
        ax.set_ylim([0, (size - 1)])
        ax.set_zlim([0, (size - 1)])
        ax.set_xlabel("Red", labelpad=-10)
        ax.set_ylabel("Green", labelpad=-10)
        ax.set_zlabel("Blue", labelpad=-12, rotation=90)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_title(title)

    fig.subplots_adjust(
        hspace=0.25, wspace=0.15, top=0.95, bottom=0.05, left=0.05, right=0.95
    )
    return fig
