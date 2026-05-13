# quiver3d.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

    
def plot_3d_quiver(
    points,
    derivatives,
    points_color=None,
    s=20,
    alpha=0.3,
    arrow_size=0.2,
    normalize=True,
    title="",
    cmap="coolwarm",
    show_colorbar=True,
    show_axes=True
):
    x, y, z = points.T
    dx, dy, dz = derivatives.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)

    if points_color is None:
        points_color = "gray"
        arrow_colors = "gray"
    else:
        cmap_obj = cm.get_cmap(cmap)
        norm = mcolors.Normalize(
            vmin=np.min(points_color),
            vmax=np.max(points_color),
        )
        arrow_colors = cmap_obj(norm(points_color))

    # --- scatter ---
    col = ax.scatter(
        x, y, z,
        c=points_color,
        s=s,
        alpha=alpha,
        cmap=None if isinstance(points_color, str) else cmap_obj,
        norm=None if isinstance(points_color, str) else norm,
    )

    # --- quiver ---
    ax.quiver(
        x, y, z,
        dx, dy, dz,
        length=arrow_size,
        normalize=normalize,
        color=arrow_colors,
        alpha=1.0,
        arrow_length_ratio=0.6,
    )

    ax.view_init(azim=-60, elev=9)

    # --------------------------------------------------
    # AXIS CLEANUP (this is the key part)
    # --------------------------------------------------
    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        # remove panes (the grey boxes)
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # remove grid lines
        ax.grid(False)

    if show_colorbar and not isinstance(points_color, str):
        fig.colorbar(
            col,
            ax=ax,
            orientation="horizontal",
            shrink=0.6,
            aspect=60,
            pad=0.01,
            label="Point Color",
        )

    fig.canvas.draw_idle()
    return fig
