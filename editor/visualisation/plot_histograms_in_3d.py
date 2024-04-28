from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import ceil
from typing import Dict
import numpy as np


def plot_histograms_in_3d(
    histograms: Dict[str, np.ndarray], histogram_per_row: int = 3
):
    cols = min(histogram_per_row, len(histograms))
    rows = ceil(len(histograms) / histogram_per_row)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)],
    )
    for i, (title, histogram) in enumerate(histograms.items()):
        fig.add_trace(
            _get_3d_scatter_plot_from_histogram(title, histogram),
            row=(i // (histogram_per_row + 1)) + 1,
            col=(i % histogram_per_row) + 1,
        )
    fig.show()


def _get_3d_scatter_plot_from_histogram(title, histogram):
    x, y, z, marker_size = [], [], [], []
    bins = len(histogram)

    for i, row in enumerate(histogram):
        for j, col in enumerate(row):
            for k, value in enumerate(col):
                if value > 0:
                    x.append(i)
                    y.append(j)
                    z.append(k)
                    marker_size.append(value)

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=[min(20, ms * 10000) for ms in marker_size],
            color=[
                f"rgb({xi*256/bins},{yi*256/bins},{zi*256/bins})"
                for xi, yi, zi in zip(x, y, z)
            ],
            opacity=0.8,
        ),
        name=title,
    )
