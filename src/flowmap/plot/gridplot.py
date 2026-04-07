import numpy as np
import matplotlib.pyplot as plt

from flowmap.core.thin_plate_spline import ThinPlateSpline
from flowmap.utils import compute_velocity_on_grid


def plot_velocity_grid(
    X_emb,
    spline_vf=None,
    V=None,
    grid_size=50,
    grid_density=1.0,
    smooth=0.5,
    n_neighbors=None,
    min_mass=0.01,
    scatter_color=None,
    scatter_size=80,
    scatter_alpha=0.1,
    arrow_color="black",
    arrow_alpha=0.9,
    arrow_scale=3.0,
    arrow_width=0.0025,
    figsize=(6, 6),
    cmap="viridis",
    vmin=None,
    vmax=None,
    show_axes=False,
    show_colorbar=False,
):
    """
    Plot a 2D velocity field as quiver arrows evaluated on a grid.

    This visualization evaluates a smooth vector field on a masked
    rectilinear grid and displays it using matplotlib's quiver plot.

    Parameters
    ----------
    X_emb : (n, 2) array
        2D embedding coordinates.

    spline_vf : Spline, optional
        Pre-fitted flowmap.spline.Spline model.
        If None, a spline is fitted from V.

    V : (n, 2) array, optional
        Raw velocity vectors corresponding to X_emb.
        Required if `spline_vf` is None.

    grid_size : int
        Resolution of the evaluation grid.

    grid_density : float
        Multiplier controlling grid sampling density.

    smooth : float
        Density smoothing parameter used for grid masking.

    min_mass : float
        Threshold for masking low-density grid regions.

    arrow_scale : float
        Scaling factor controlling quiver arrow length.

    arrow_width : float
        Width of quiver arrows.

    Notes
    -----
    - Grid construction and masking are handled by
      `flowmap.utils.compute_velocity_on_grid`.
    - Designed for RNA velocity visualization but applicable
      to any 2D vector field.
    """

    # --------------------------------------------------
    # 1. Fit spline if needed
    # --------------------------------------------------
    if spline_vf is None:
        if V is None:
            raise ValueError("Either `spline_vf` or `V` must be provided.")

        n_points = X_emb.shape[0]
        max_points = 4000

        if n_points > max_points:
            idx = np.random.choice(n_points, max_points, replace=False)
            X_fit = X_emb[idx]
            V_fit = V[idx]
            print(f"[Spline] Subsampling {max_points}/{n_points} points for fitting …")
        else:
            X_fit = X_emb
            V_fit = V
            print(f"[Spline] Using all {n_points} points for fitting …")

        spline = ThinPlateSpline(X_fit)
        spline.fit(V_fit)

    # --------------------------------------------------
    # 2. Evaluate on grid
    # --------------------------------------------------
    Xg, keep, Vg, _ = compute_velocity_on_grid(
        X_emb,
        spline_vf=spline_vf,
        grid_size=grid_size,
        grid_density=grid_density,
        smooth=smooth,
        n_neighbors=n_neighbors,
        min_mass=min_mass,
        return_mesh=True,
    )

    # --------------------------------------------------
    # 3. Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # ---- scatter background ----
    if scatter_color is None:
        ax.scatter(
            X_emb[:, 0],
            X_emb[:, 1],
            s=scatter_size,
            color="gray",
            alpha=scatter_alpha,
        )
    else:
        scatter_color = np.asarray(scatter_color)

        if np.issubdtype(scatter_color.dtype, np.number):
            sc = ax.scatter(
                X_emb[:, 0],
                X_emb[:, 1],
                s=scatter_size,
                c=scatter_color,
                alpha=scatter_alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if show_colorbar:
                fig.colorbar(sc, ax=ax)
        else:
            unique = np.unique(scatter_color)
            cmap_obj = plt.get_cmap(cmap, len(unique))
            lut = {k: cmap_obj(i) for i, k in enumerate(unique)}
            mapped = np.array([lut[v] for v in scatter_color])

            ax.scatter(
                X_emb[:, 0],
                X_emb[:, 1],
                s=scatter_size,
                color=mapped,
                alpha=scatter_alpha,
            )

    # ---- quiver arrows ----
    ax.quiver(
        Xg[:, 0],
        Xg[:, 1],
        Vg[:, 0],
        Vg[:, 1],
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
        width=arrow_width,
        headwidth=3,
        color=arrow_color,
        alpha=arrow_alpha,
    )

    ax.set_aspect("equal")

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    else:
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    fig.canvas.draw_idle()
    return fig

