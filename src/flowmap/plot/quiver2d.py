# quiver2d.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm


def plot_2d_quiver(
    X,
    V,
    color,
    scale=1.0,
    normalize=False,
    size=10,
    alpha=0.5,
    cmap="coolwarm",
    title="",
):
    if normalize:
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V = V / norms

    cmap_obj = cm.get_cmap(cmap)
    norm = mcolors.Normalize(
        vmin=np.min(color),
        vmax=np.max(color),
    )

    rgba = cmap_obj(norm(color))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)

    ax.scatter(
        X[:, 0],
        X[:, 1],
        color=rgba,
        s=size,
        alpha=alpha,
        edgecolors="none",
    )

    ax.quiver(
        X[:, 0],
        X[:, 1],
        V[:, 0],
        V[:, 1],
        color=rgba,
        scale=scale,
        angles="xy",
        scale_units="xy",
        width=0.003,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    fig.canvas.draw_idle()
    return fig

