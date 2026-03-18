import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal

def compute_velocity_on_grid(
    X_emb,
    spline_vf=None,
    *,
    grid_size=100,
    grid_density=1.0,
    smooth=0.5,
    n_neighbors=None,
    min_mass=0.01,
    margin_ratio=0.0,
    return_mesh=False,
):
    """
    Internal utility for constructing a masked evaluation grid
    over an embedding and optionally evaluating a spline-based vector field.
    """

    idx_valid = np.isfinite(X_emb).all(axis=1)
    X_emb = X_emb[idx_valid]

    if X_emb.size == 0:
        raise ValueError("No valid points.")

    n_obs, d = X_emb.shape

    if n_neighbors is None:
        n_neighbors = max(10, int(n_obs / 30))

    # --- define grid bounds ---
    bounds = []
    grids = []

    for i in range(d):
        lo, hi = np.min(X_emb[:, i]), np.max(X_emb[:, i])
        pad = 0.01 * (hi - lo + 1e-5)
        lo, hi = lo - pad, hi + pad
        bounds.append((lo, hi))
        grids.append(np.linspace(lo, hi, int(grid_size * grid_density)))

    meshes = np.meshgrid(*grids, indexing="xy")
    X_grid = np.vstack([g.ravel() for g in meshes]).T

    # --- density weighting ---
    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(X_emb)
    dists, _ = nn.kneighbors(X_grid)

    scale = np.mean([(g[1] - g[0]) for g in grids]) * smooth
    p_mass = normal.pdf(dists, scale=scale).sum(1)

    keep = p_mass > (np.percentile(p_mass, 99) * min_mass)
    Xg = X_grid[keep]

    # --- optional boundary cropping ---
    if margin_ratio > 0:
        mask_margin = np.ones(len(Xg), dtype=bool)
        for i in range(d):
            lo, hi = bounds[i]
            margin = margin_ratio * (hi - lo)
            mask_margin &= (
                (Xg[:, i] > lo + margin) &
                (Xg[:, i] < hi - margin)
            )
        Xg = Xg[mask_margin]

    # --- optional velocity prediction ---
    Vg = spline_vf.predict(Xg) if spline_vf is not None else None

    if return_mesh:
        return Xg, keep, Vg, meshes

    return Xg, keep, Vg

