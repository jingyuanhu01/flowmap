import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from flowmap.core.thin_plate_spline import ThinPlateSpline
from flowmap.utils import compute_velocity_on_grid


def plot_velocity_stream(
    X_2d,
    spline=None,
    V=None,
    scatter_color="grey",
    grid_size=50,
    grid_density=1.0,
    stream_density=1.0,
    title=None,
    scatter_size=10,
    scatter_alpha=0.5,
    arrowsize=1.5,
    ax=None,
    figsize=(8, 6),
    aspect="equal",
    cmap="tab10",
    vmin=None,
    vmax=None,
    show_axes=False,
    show_colorbar=False,
    streamline_thickness=4.0,
    pad_frac=0.0,
):
    """
    Plot a 2D velocity field as streamlines over an embedding.

    Parameters
    ----------
    X_2d : (n, 2) array
        2D embedding coordinates.

    spline : Spline, optional
        Pre-fitted flowmap.spline.Spline model mapping X_2d → velocity.
        If None, a spline will be fitted using V.

    V : (n, 2) array, optional
        Raw velocity vectors. Required if `spline` is None.

    grid_size : int
        Resolution of evaluation grid.

    stream_density : float
        Density of streamlines.

    streamline_thickness : float
        Scaling factor for linewidth proportional to speed magnitude.

    pad_frac : float
        Fractional padding added around axis limits.
    """

    # --------------------------------------------------
    # 1. Fit spline if not provided
    # --------------------------------------------------
    if spline is None:
        if V is None:
            raise ValueError("Either `spline` or `V` must be provided.")

        n_points = X_2d.shape[0]
        max_points = 4000

        if n_points > max_points:
            idx = np.random.choice(n_points, max_points, replace=False)
            X_fit = X_2d[idx]
            V_fit = V[idx]
            print(f"[Spline] Subsampling {max_points}/{n_points} points for fitting …")
        else:
            X_fit = X_2d
            V_fit = V
            print(f"[Spline] Using all {n_points} points for fitting …")

        spline = ThinPlateSpline(X_fit)
        spline.fit(V_fit)

    # --------------------------------------------------
    # 2. Evaluate field on grid
    # --------------------------------------------------
    Xg, keep, Vg, (xx, yy) = compute_velocity_on_grid(
        X_2d,
        spline_vf=spline,
        grid_size=grid_size,
        grid_density=grid_density,
        min_mass=0.01,
        return_mesh=True,
    )

    # --------------------------------------------------
    # 3. Reconstruct dense grid for streamplot
    # --------------------------------------------------
    ny, nx = yy.shape[0], xx.shape[1]
    grid_x = xx[0, :]
    grid_y = yy[:, 0]

    Vx = np.full((ny, nx), np.nan)
    Vy = np.full((ny, nx), np.nan)

    dx = (grid_x[-1] - grid_x[0]) / (nx - 1)
    dy = (grid_y[-1] - grid_y[0]) / (ny - 1)

    j_idx = np.clip(
        np.rint((Xg[:, 0] - grid_x[0]) / dx).astype(int),
        0,
        nx - 1,
    )
    i_idx = np.clip(
        np.rint((Xg[:, 1] - grid_y[0]) / dy).astype(int),
        0,
        ny - 1,
    )

    Vx[i_idx, j_idx] = Vg[:, 0]
    Vy[i_idx, j_idx] = Vg[:, 1]

    U = np.ma.masked_invalid(Vx)
    W = np.ma.masked_invalid(Vy)

    # --------------------------------------------------
    # 4. Plot
    # --------------------------------------------------
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # ---- scatter background ----
    scatter_color = np.asarray(scatter_color)

    if scatter_color.ndim == 0:
        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            s=scatter_size,
            alpha=scatter_alpha,
            color=scatter_color,
            edgecolors="none",
        )

    elif np.issubdtype(scatter_color.dtype, np.number):
        sc = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            s=scatter_size,
            alpha=scatter_alpha,
            c=scatter_color,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )
        if show_colorbar:
            plt.colorbar(sc, ax=ax)

    else:
        unique_vals = np.unique(scatter_color)
        cmap_obj = plt.get_cmap(cmap, len(unique_vals))
        lut = {val: cmap_obj(i) for i, val in enumerate(unique_vals)}
        mapped = np.array([lut[v] for v in scatter_color])

        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            s=scatter_size,
            alpha=scatter_alpha,
            color=mapped,
            edgecolors="none",
        )

    # ---- streamlines ----
    speed = np.sqrt(Vx**2 + Vy**2)
    smax = np.nanmax(speed) if np.isfinite(speed).any() else 0.0
    linewidth = (
        streamline_thickness * (speed / smax)
        if smax > 0
        else 1.0
    )

    ax.streamplot(
        grid_x,
        grid_y,
        U,
        W,
        linewidth=linewidth,
        density=stream_density,
        color="k",
        arrowsize=arrowsize,
        arrowstyle="-|>",
        maxlength=4,
        integration_direction="both",
    )

    # ---- axis formatting ----
    if pad_frac > 0:
        xmin, xmax = X_2d[:, 0].min(), X_2d[:, 0].max()
        ymin, ymax = X_2d[:, 1].min(), X_2d[:, 1].max()

        dx = xmax - xmin
        dy = ymax - ymin

        ax.set_xlim(xmin - pad_frac * dx, xmax + pad_frac * dx)
        ax.set_ylim(ymin - pad_frac * dy, ymax + pad_frac * dy)

    ax.set_aspect(aspect)

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    else:
        ax.grid(True, linestyle="--", alpha=0.3)

    if title:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()

    fig = ax.figure
    fig.canvas.draw_idle()   # ensures Jupyter renders it

    return fig
