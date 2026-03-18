# flowmap/geometry/fixed_points.py

from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq, eigvals
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Optional
from flowmap.utils import compute_velocity_on_grid


class FixedPointAnalyzer:
    """
    Fixed-point detection and classification for smooth vector fields
    defined on a FlowMap embedding.

    Parameters
    ----------
    embedding : VectorFieldEmbedder
        Must expose:

        - X_emb : ndarray (N, d)
        - V_emb : ndarray (N, d)
        - spline : Spline
        - spline_vf : Spline
    """

    def __init__(self, embedding):

        self.embedding = embedding

        self.X_emb = embedding.X_emb
        self.V_emb = embedding.V_emb
        self.spline = embedding.spline
        self.spline_vf = embedding.spline_vf

        self.fixed_point_info: List[Dict] = []


    def identify_fixed_points(
        self,
        *,
        grid_resolution: int = 100,
        speed_smoothing: float = 1.0,
        speed_quantile_threshold: float = 0.1,
        jacobian_radius: float = 0.2,
        weighted_jacobian: bool = True,
        **grid_kwargs,
    ):
        """
        Identify and classify fixed points of the vector field in embedding space.

        This method follows a simple pipeline:

        1. **Evaluate velocity on a grid**
           A regular grid is constructed over the embedding and the vector field
           is evaluated using the spline model.

        2. **Compute speed**
           The metric-aware speed is computed at each grid point:

               ||v(x)||_g = sqrt(v(x)^T G(x) v(x))

        3. **Smooth the speed field**
           A Gaussian filter is applied to reduce noise and stabilize minima detection.

        4. **Detect candidate fixed points**
           Local minima of the smoothed speed field are identified.
           Only points with sufficiently low speed (based on a global quantile)
           are kept as candidates.

        5. **Refine and classify**
           Each candidate is optionally refined by minimizing ||v(x)||², and
           classified using the eigenvalues of a locally fitted Jacobian.

        Parameters
        ----------
        grid_resolution : int, default=100
            Number of grid points per dimension used to evaluate the vector field.
            Higher values give finer resolution but increase computation.

        speed_smoothing : float, default=1.0
            Standard deviation of Gaussian smoothing applied to the speed field.
            Helps remove noise before detecting minima.

        speed_quantile_threshold : float, default=0.1
            Quantile threshold for selecting low-speed candidates.
            For example, 0.1 keeps the lowest 10% of speeds.

        jacobian_radius : float, default=0.2
            Radius (relative to embedding scale) used to select nearby points
            for local Jacobian estimation.

        weighted_jacobian : bool, default=True
            If True, nearby points are weighted more strongly when fitting
            the Jacobian.

        **grid_kwargs
            Additional arguments passed to `compute_velocity_on_grid`, such as:

            - n_neighbors
            - min_mass
            - smooth (density smoothing)
            - margin_ratio

        Returns
        -------
        list of dict
            Each dictionary contains:

            - ``position`` : ndarray of shape (d,)
            - ``jacobian`` : ndarray of shape (d, d)
            - ``type`` : str

        Notes
        -----
        - Fixed points correspond to locations where velocity magnitude is minimal.
        - Stability is determined from the eigenvalues of the local Jacobian.
        """

        # --------------------------------------------------------------
        # 0) Evaluate vector field on grid
        # --------------------------------------------------------------
        grid_points, grid_mask, grid_velocity = compute_velocity_on_grid(
            self.X_emb,
            spline_vf=self.spline_vf,
            grid_size=grid_resolution,
            **grid_kwargs,
        )

        if grid_points is None or len(grid_points) == 0:
            return []

        # --------------------------------------------------------------
        # Infer grid shape (2D only)
        # --------------------------------------------------------------
        d = self.X_emb.shape[1]
        if d != 2:
            raise ValueError(
                "FixedPointAnalyzer currently expects a 2D embedding grid for minima detection. "
                f"Got d={d}."
            )

        n_full = int(np.asarray(grid_mask).shape[0])
        side = int(np.round(np.sqrt(n_full)))
        if side * side != n_full:
            raise ValueError(
                "Cannot infer 2D grid_shape from mask. "
                f"Expected square full grid in 2D; got n_full={n_full}."
            )
        grid_shape = (side, side)

        # --------------------------------------------------------------
        # 1) Compute metric speed on grid
        # --------------------------------------------------------------
        metric_tensor = self.spline.compute_metric(grid_points)
        speed_values = np.sqrt(
            np.einsum("ni,nij,nj->n", grid_velocity, metric_tensor, grid_velocity)
        )

        # Map back to full grid (NaN outside valid region)
        speed_grid = np.full(grid_shape, np.nan, dtype=float)
        speed_grid.ravel()[grid_mask] = speed_values

        filled_speed_grid = np.nan_to_num(
            speed_grid,
            nan=np.nanmax(speed_grid),
        )

        # --------------------------------------------------------------
        # 2) Smooth speed field
        # --------------------------------------------------------------
        smoothed_speed = gaussian_filter(
            filled_speed_grid,
            sigma=speed_smoothing,
        )

        # --------------------------------------------------------------
        # 3) Detect local minima
        # --------------------------------------------------------------
        minima_mask = self._local_minima_2d(smoothed_speed)
        minima_indices = np.argwhere(minima_mask)

        if len(minima_indices) == 0:
            return []

        # --------------------------------------------------------------
        # 4) Filter by low speed (global threshold)
        # --------------------------------------------------------------
        speed_threshold = np.nanquantile(speed_values, speed_quantile_threshold)

        flat_indices = np.ravel_multi_index(
            (minima_indices[:, 0], minima_indices[:, 1]),
            dims=grid_shape,
        )

        # Keep only valid grid points
        valid_mask = grid_mask[flat_indices]
        flat_indices = flat_indices[valid_mask]
        if len(flat_indices) == 0:
            return []

        # Keep only low-speed candidates
        low_speed_mask = speed_grid.ravel()[flat_indices] <= speed_threshold
        flat_indices = flat_indices[low_speed_mask]
        if len(flat_indices) == 0:
            return []

        # Map back to coordinates
        valid_indices = np.flatnonzero(grid_mask)
        lookup = {idx: i for i, idx in enumerate(valid_indices)}
        candidate_points = grid_points[[lookup[i] for i in flat_indices]]

        # --------------------------------------------------------------
        # 5) Optional refinement
        # --------------------------------------------------------------
        refined_points = self._refine_candidates(candidate_points)

        # --------------------------------------------------------------
        # 6) Local Jacobian classification
        # --------------------------------------------------------------
        results = []
        for point in refined_points:
            info = self._classify_point(
                point,
                radius_percent=jacobian_radius,
                weighted=weighted_jacobian,
            )
            if info is not None:
                results.append(info)

        self.fixed_point_info = results
        return results

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _local_minima_2d(Z: np.ndarray) -> np.ndarray:
        mask = np.ones_like(Z, dtype=bool)

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                mask &= Z < np.roll(np.roll(Z, di, 0), dj, 1)

        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False

        return mask


    def _refine_candidates(self, fps: np.ndarray) -> np.ndarray:
        """
        Refine fixed-point candidates by minimizing ||v(x)||² locally.
        """

        def vf_energy(x):
            v = self.spline_vf.predict(x[None, :])[0]
            return np.dot(v, v)

        refined = []

        for fp in fps:
            try:
                res = minimize(
                    vf_energy,
                    x0=fp,
                    method="L-BFGS-B",
                    options=dict(maxiter=15),
                )
                refined.append(res.x if res.success else fp)
            except Exception:
                refined.append(fp)

        return np.asarray(refined)


    def _classify_point(self, fp, *, radius_percent, weighted):

        emb_range = np.ptp(self.X_emb, axis=0)
        radius = radius_percent * np.mean(emb_range)

        dx = self.X_emb - fp
        mask = np.linalg.norm(dx, axis=1) < radius

        if mask.sum() < 10:
            return None

        Xloc = dx[mask]
        Vloc = self.V_emb[mask]

        J = self._fit_jacobian(Xloc, Vloc, weighted=weighted)
        fp_type = self.classify_fixed_point(J)

        return dict(
            position=fp,
            jacobian=J,
            type=fp_type,
        )


    @staticmethod
    def _fit_jacobian(Xloc, Vloc, weighted=True):

        if weighted:
            sigma = 0.5 * np.mean(np.ptp(Xloc, axis=0))
            w = np.exp(-np.sum(Xloc**2, axis=1) / (2 * sigma**2))[:, None]
            Xloc = Xloc * w
            Vloc = Vloc * w

        J, *_ = lstsq(Xloc, Vloc, rcond=None)
        return J


    # ------------------------------------------------------------------
    # Linear stability classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_fixed_point(J: np.ndarray, eps: float = 1e-4) -> str:
        """
        Classify a fixed point using linear stability analysis of its Jacobian.
    
        The classification is determined from the eigenvalue spectrum
        of the Jacobian matrix evaluated at the fixed point.
    
        Parameters
        ----------
        J : ndarray of shape (d, d)
            Jacobian matrix of the vector field at the fixed point.
        eps : float, default=1e-4
            Numerical tolerance used to determine whether eigenvalues
            are considered zero or have non-negligible imaginary parts.
    
        Returns
        -------
        str
            One of:
    
            - "Stable node (sink)"
            - "Unstable node (source)"
            - "Saddle"
            - "Degenerate"
            - "Center"
            - "Spiral in"
            - "Spiral out"
            - "Mixed focus"
    
        Notes
        -----
        Let :math:`\\lambda_i \\in \\mathbb{C}` denote the eigenvalues of ``J``.
    
        - Stability is determined by the sign of :math:`\\Re(\\lambda_i)`.
        - Purely real spectra correspond to node or saddle behavior.
        - Complex conjugate pairs indicate rotational (spiral or center) dynamics.
        - A fixed point is:
            - Stable if all :math:`\\Re(\\lambda_i) < 0`
            - Unstable if all :math:`\\Re(\\lambda_i) > 0`
            - A saddle if signs are mixed
    
        This classification corresponds to the local linearization of the
        dynamical system near the fixed point.
        """

        eigs = eigvals(J)
        re, im = eigs.real, eigs.imag
        has_im = np.any(np.abs(im) > eps)

        if not has_im:
            if np.all(re < -eps): return "Stable node (sink)"
            if np.all(re >  eps): return "Unstable node (source)"
            if np.any(re > eps) and np.any(re < -eps): return "Saddle"
            return "Degenerate"

        a = re[np.abs(im) > eps][0]

        if abs(a) < eps: return "Center"
        if a < 0: return "Spiral in"
        if a > 0: return "Spiral out"
        return "Mixed focus"

