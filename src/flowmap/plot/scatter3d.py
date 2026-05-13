# scatter3d.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_3d_scatter(points, points_color=None, title="",
            azim=-60, elev=9,
            grid_interval=2,
            dot_size=5, alpha=0.4):
    
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)

    # Set transparent plot background
    ax.set_facecolor((0, 0, 0, 0))

    # Scatter with user-defined size and transparency
    scatter_kwargs = {"color": "gray"} if points_color is None else {"c": points_color}
    col = ax.scatter(x, y, z, **scatter_kwargs,
                     s=dot_size, alpha=alpha,
                     edgecolors='none')

    # View angle
    ax.view_init(azim=azim, elev=elev)

    # Sparse grid, no tick labels
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(ticker.MultipleLocator(grid_interval))
        axis.set_ticklabels([])

    ax.grid(True)

    # Colorbar
    if points_color is not None:
        fig.colorbar(col, ax=ax,
                     orientation="horizontal",
                     shrink=0.6, aspect=60, pad=0.01)

    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig
