import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def plot_histograms_in_2d(histograms: Dict[str, np.ndarray], figsize=(15, 5)):
    fig = plt.figure(figsize=figsize)

    for i, (title, histogram) in enumerate(histograms.items(), 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")

        size = histogram.shape[0]

        x, y, z = np.indices(histogram.shape)
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = histogram.flatten()

        sizes = values * 5000

        colors = np.vstack((x, y, z)).T / (size - 1)

        sc = ax.scatter(x, y, z, c=colors, s=sizes, marker="o", alpha=0.5)

        ax.set_xlim([0, (size - 1)])
        ax.set_ylim([0, (size - 1)])
        ax.set_zlim([0, (size - 1)])
        ax.set_title(title)

    return fig
