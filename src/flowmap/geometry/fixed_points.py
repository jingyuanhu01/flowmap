# flowmap/geometry/fixed_points.py

from __future__ import annotations

import numpy as np
from numpy.linalg import eigvals
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, minimum_filter
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
        mode: str = "euclidean",
        grid_resolution: int = 60,
        speed_smoothing: float = 1.0,
        speed_quantile_threshold: float = 0.1,
        max_candidates: int = 20,
        min_separation: Optional[float] = None,
        refine: bool = True,
        metric_regularization: float = 1e-8,
        jacobian_radius: float = 0.15,
        weighted_jacobian: bool = True,
        **grid_kwargs,
    ):
        """
        Identify and classify fixed points of the vector field in embedding space.

        Two analysis modes are supported:

        ``mode="euclidean"``
            Treat the embedding as an ordinary Euclidean plane. Candidate fixed
            points are low-speed local minima of ``||nu(y)||`` and stability is
            classified from the embedding velocity Jacobian ``J_nu(y*)``.

        ``mode="riemannian"``
            Find fixed points as zeros of the embedding velocity ``nu(y)``,
            then use the pullback metric for local classification. Near a fixed
            point ``y*``, the metric ``g(y*) = L L^T`` defines flattened
            coordinates ``z = L^T (y - y*)``. Stability is classified from

            ``A = L^T J_nu(y*) L^{-T}``.

            This matches the local flattening/linearization described in the
            manuscript.

        The detection pipeline is:

        1. Evaluate the fitted embedding velocity spline on a regular grid.
        2. Find low-speed candidates from the Euclidean embedding speed
           ``||nu(y)||``. Fixed points are zeros of ``nu`` in both modes.
        3. Smooth the speed field and select low-speed local minima.
        4. Optionally refine candidates by minimizing ``||nu(y)||²``.
        5. Deduplicate nearby candidates and classify the local linear system.

        Parameters
        ----------
        mode : {"euclidean", "riemannian"}, default="euclidean"
            Fixed-point analysis mode.

            - ``"euclidean"`` ignores the metric entirely.
            - ``"riemannian"`` uses the pullback metric and local flattening.

        grid_resolution : int, default=100
            Number of grid points per dimension used to evaluate the vector field.
            Higher values give finer resolution but increase computation.

        speed_smoothing : float, default=1.0
            Standard deviation of Gaussian smoothing applied to the speed field.
            Helps remove noise before detecting minima.

        speed_quantile_threshold : float, default=0.1
            Quantile threshold for selecting low-speed candidates.
            For example, 0.1 keeps the lowest 10% of speeds.

        max_candidates : int, default=20
            Maximum number of low-speed candidates to refine and classify.

        min_separation : float, optional
            Minimum distance between returned fixed points in embedding units.
            If None, a grid-scale separation is used.

        refine : bool, default=True
            If True, locally refine candidates by minimizing the mode-specific
            squared speed.

        metric_regularization : float, default=1e-8
            Diagonal regularization added before Cholesky factorization in
            Riemannian mode.

        jacobian_radius : float, default=0.2
            Deprecated. Kept for compatibility with older notebooks.

        weighted_jacobian : bool, default=True
            Deprecated. Kept for compatibility with older notebooks.

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
            - ``jacobian`` : embedding velocity Jacobian ``J_nu(y*)``
            - ``linearization`` : matrix used for stability classification
            - ``type`` : str
            - ``speed`` : Euclidean embedding speed at the candidate
            - ``metric_speed`` : Riemannian speed, in Riemannian mode
            - ``mode`` : analysis mode
        """
        del jacobian_radius, weighted_jacobian

        mode = mode.lower()
        if mode not in {"euclidean", "riemannian"}:
            raise ValueError("mode must be 'euclidean' or 'riemannian'.")

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
        # 1) Compute Euclidean embedding speed on grid.
        #
        # Fixed points are zeros of dy/dt = nu(y), independent of the metric.
        # The metric is used in riemannian mode for local flattening and
        # stability classification, not for deciding where nu(y) is zero.
        # --------------------------------------------------------------
        speed_values = self._speed(grid_points, grid_velocity, mode=mode)

        # Map back to full grid (NaN outside valid region)
        speed_grid = np.full(grid_shape, np.nan, dtype=float)
        speed_grid.ravel()[grid_mask] = speed_values

        if not np.isfinite(speed_values).any():
            return []

        filled_speed_grid = np.nan_to_num(
            speed_grid,
            nan=np.nanmax(speed_values),
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

        # --------------------------------------------------------------
        # 4) Filter by low speed (global threshold)
        # --------------------------------------------------------------
        speed_threshold = np.nanquantile(speed_values, speed_quantile_threshold)

        flat_indices = np.flatnonzero(minima_mask.ravel())

        # Keep only valid grid points
        valid_mask = grid_mask[flat_indices]
        flat_indices = flat_indices[valid_mask]

        # Keep only low-speed candidates
        if len(flat_indices) > 0:
            low_speed_mask = speed_grid.ravel()[flat_indices] <= speed_threshold
            flat_indices = flat_indices[low_speed_mask]

        # Robust fallback: if smoothing/local-minimum detection misses a flat
        # basin, seed refinement from the globally slowest valid grid points.
        if len(flat_indices) == 0:
            valid_indices = np.flatnonzero(grid_mask)
            order = np.argsort(speed_grid.ravel()[valid_indices])
            flat_indices = valid_indices[order[:max_candidates]]

        # Map back to coordinates
        valid_indices = np.flatnonzero(grid_mask)
        lookup = {idx: i for i, idx in enumerate(valid_indices)}
        flat_indices = np.asarray(flat_indices)
        flat_indices = flat_indices[
            np.argsort(speed_grid.ravel()[flat_indices])
        ][:max_candidates]
        candidate_points = grid_points[[lookup[i] for i in flat_indices]]

        # --------------------------------------------------------------
        # 5) Optional refinement
        # --------------------------------------------------------------
        if refine:
            refined_points = self._refine_candidates(
                candidate_points,
                mode="euclidean",
                metric_regularization=metric_regularization,
            )
        else:
            refined_points = candidate_points

        refined_points = self._dedupe_points(
            refined_points,
            mode="euclidean",
            min_separation=min_separation,
        )

        # --------------------------------------------------------------
        # 6) Local linearization and classification
        # --------------------------------------------------------------
        results = []
        for point in refined_points:
            info = self._classify_point(
                point,
                mode=mode,
                metric_regularization=metric_regularization,
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
        local_min = minimum_filter(Z, size=3, mode="nearest")
        mask = Z <= local_min
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
        return mask


    def _speed(
        self,
        points: np.ndarray,
        velocities: np.ndarray,
        *,
        mode: str,
    ) -> np.ndarray:
        if mode == "euclidean":
            return np.linalg.norm(velocities, axis=1)

        metric = self.spline.compute_metric(points)
        speed2 = np.einsum("ni,nij,nj->n", velocities, metric, velocities)
        return np.sqrt(np.maximum(speed2, 0.0))


    def _energy(
        self,
        point: np.ndarray,
        *,
        mode: str,
        metric_regularization: float,
    ) -> float:
        point = np.asarray(point, dtype=float)
        v = self._predict_velocity(point)

        if mode == "euclidean":
            return float(np.dot(v, v))

        g = self._regularized_metric(point, metric_regularization)
        return float(v @ g @ v)


    def _refine_candidates(
        self,
        fps: np.ndarray,
        *,
        mode: str,
        metric_regularization: float,
    ) -> np.ndarray:
        """
        Refine fixed-point candidates by minimizing mode-specific speed².
        """

        def vf_energy(x):
            return self._energy(
                x,
                mode=mode,
                metric_regularization=metric_regularization,
            )

        margin = 0.05 * np.ptp(self.X_emb, axis=0)
        bounds = [
            (lo - pad, hi + pad)
            for lo, hi, pad in zip(
                np.min(self.X_emb, axis=0),
                np.max(self.X_emb, axis=0),
                margin,
            )
        ]

        refined = []

        for fp in fps:
            try:
                res = minimize(
                    vf_energy,
                    x0=fp,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=dict(maxiter=50),
                )
                refined.append(res.x if res.success else fp)
            except Exception:
                refined.append(fp)

        return np.asarray(refined)


    def _dedupe_points(
        self,
        points: np.ndarray,
        *,
        mode: str,
        min_separation: Optional[float],
    ) -> np.ndarray:
        if len(points) == 0:
            return points

        if min_separation is None:
            min_separation = 1.5 * np.mean(np.ptp(self.X_emb, axis=0)) / 100.0

        order = np.argsort([
            self._energy(p, mode=mode, metric_regularization=1e-8)
            for p in points
        ])

        kept = []
        for idx in order:
            point = points[idx]
            if all(np.linalg.norm(point - prev) >= min_separation for prev in kept):
                kept.append(point)

        return np.asarray(kept)


    def _classify_point(
        self,
        fp: np.ndarray,
        *,
        mode: str,
        metric_regularization: float,
    ):
        jacobian = self._velocity_jacobian(fp)

        if mode == "euclidean":
            linearization = jacobian
            metric = None
        else:
            metric = self._regularized_metric(fp, metric_regularization)
            linearization = self._flattened_linearization(jacobian, metric)

        fp_type = self.classify_fixed_point(linearization)
        speed = np.sqrt(
            self._energy(
                fp,
                mode="euclidean",
                metric_regularization=metric_regularization,
            )
        )
        metric_speed = None
        if mode == "riemannian":
            metric_speed = np.sqrt(
                self._energy(
                    fp,
                    mode="riemannian",
                    metric_regularization=metric_regularization,
                )
            )

        return dict(
            position=fp,
            jacobian=jacobian,
            linearization=linearization,
            metric=metric,
            eigenvalues=eigvals(linearization),
            type=fp_type,
            speed=float(speed),
            metric_speed=None if metric_speed is None else float(metric_speed),
            mode=mode,
        )


    def _velocity_jacobian(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)

        if hasattr(self.spline_vf, "compute_jacobians"):
            return np.asarray(
                self.spline_vf.compute_jacobians(point[None, :])[0],
                dtype=float,
            )

        # Conservative fallback for spline-like objects without an analytic
        # Jacobian. This keeps the fixed-point analyzer usable in tests and
        # simple external integrations.
        d = point.shape[0]
        scale = np.mean(np.ptp(self.X_emb, axis=0))
        h = max(scale * 1e-5, 1e-6)
        jac = np.zeros((d, d), dtype=float)

        for j in range(d):
            step = np.zeros(d)
            step[j] = h
            vp = self._predict_velocity(point + step)
            vm = self._predict_velocity(point - step)
            jac[:, j] = (vp - vm) / (2 * h)

        return jac


    def _predict_velocity(self, point: np.ndarray) -> np.ndarray:
        pred = np.asarray(self.spline_vf.predict(np.asarray(point)[None, :]))
        if pred.ndim == 1:
            return pred.astype(float)
        return pred[0].astype(float)


    def _regularized_metric(
        self,
        point: np.ndarray,
        metric_regularization: float,
    ) -> np.ndarray:
        metric = np.asarray(self.spline.compute_metric(np.asarray(point)[None, :]))
        if metric.ndim == 3:
            metric = metric[0]
        metric = np.asarray(metric, dtype=float)
        metric = 0.5 * (metric + metric.T)
        metric = metric + metric_regularization * np.eye(metric.shape[0])

        eig, vec = np.linalg.eigh(metric)
        eig = np.maximum(eig, metric_regularization)
        return (vec * eig) @ vec.T


    @staticmethod
    def _flattened_linearization(
        jacobian: np.ndarray,
        metric: np.ndarray,
    ) -> np.ndarray:
        L = np.linalg.cholesky(metric)
        inv_LT = np.linalg.solve(L.T, np.eye(L.shape[0]))
        return L.T @ jacobian @ inv_LT


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
