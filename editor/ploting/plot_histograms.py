from plotly.subplots import make_subplots
import plotly.graph_objects as go
from math import ceil


def plot_histograms(hists, histogram_per_row: int = 3):
    cols = min(histogram_per_row, len(hists))
    fig = make_subplots(
        rows=ceil(len(hists) / histogram_per_row),
        cols=cols,
        specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(1)],
    )
    for i, hist in enumerate(hists, start=1):
        fig.add_trace(_get_3d_scatter_plot_from_histogram(hist), row=1, col=i)

    fig.update_layout(
        showlegend=False,
        autosize=True,
        scene1=dict(xaxis_title="R", yaxis_title="G", zaxis_title="B"),
        scene2=dict(xaxis_title="R", yaxis_title="G", zaxis_title="B"),
    )
    fig.show()


def _get_3d_scatter_plot_from_histogram(hist):
    x, y, z, marker_size = [], [], [], []
    bins = len(hist)

    for i, row in enumerate(hist):
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
    )
