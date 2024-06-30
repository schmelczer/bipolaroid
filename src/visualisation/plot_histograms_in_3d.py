from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import ceil
from typing import Dict
import numpy as np


def plot_histograms_in_3d(
    histograms: Dict[str, np.ndarray], histograms_per_row: int = 3
):
    column_count = min(histograms_per_row, len(histograms))
    row_count = ceil(len(histograms) / histograms_per_row)
    fig = make_subplots(
        rows=row_count,
        cols=column_count,
        specs=[
            [{"type": "scatter3d"} for _ in range(column_count)]
            for _ in range(row_count)
        ],
    )

    for i, (title, histogram) in enumerate(histograms.items()):
        fig.add_trace(
            _get_3d_scatter_plot_from_histogram(title, histogram),
            row=(i // histograms_per_row) + 1,
            col=(i % histograms_per_row) + 1,
        )

    scenes = {
        f"scene{i}": dict(
            camera=dict(eye=dict(x=0.1, y=0, z=2)),
            xaxis=go.layout.scene.XAxis(title="Red"),
            yaxis=go.layout.scene.YAxis(title="Green"),
            zaxis=go.layout.scene.ZAxis(title="Blue"),
        )
        for i in range(1, len(histograms) + 1)
    }
    fig.update_layout(**scenes, height=300 * column_count)
    fig.update_layout()  # You can adjust the height as needed

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
            opacity=1,
            line=dict(width=0),
        ),
        name=title,
    )
