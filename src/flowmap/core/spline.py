"""
flowmap.spline
==============

Polyharmonic spline model used for geometry-aware vector field embeddings.

Implements a 2D polyharmonic spline with kernel:

    φ(r) = r⁴ log r

This choice guarantees C² continuity of the mapping and well-defined
Hessians, making it suitable for curvature and geometric computations.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import svd
from scipy.optimize import root_scalar
from sklearn.cluster import KMeans
from math import comb
from typing import Optional, Tuple


class Spline:
    """
    Polyharmonic spline regression model (2D triharmonic spline).

    This model represents a smooth mapping

        f : ℝᵈ → ℝᴰ

    of the form:

        f(x) = Σᵢ wᵢ φ(||x - cᵢ||) + P₂(x) a

    where:

    - φ(r) = r⁴ log r  (polyharmonic kernel of order m=3 in 2D)
    - cᵢ are control points
    - wᵢ are radial weights
    - P₂(x) contains all monomials up to degree 2
    - a are polynomial coefficients

    The radial weights are ridge-regularized.

    This spline is C² smooth and suitable for:

    - Jacobian computation
    - Hessian computation
    - Riemannian metric construction
    - Curvature and torsion analysis
    - Velocity pushforwards

    Parameters
    ----------
    X : ndarray of shape (N, d)
        Input training points.

    n_control_points : int, optional (default=1000)
        Number of control points. If larger than N,
        all points are used.

    max_n : int, optional (default=4000)
        Warning threshold for large datasets.

    Notes
    -----
    For polyharmonic splines of order m=3 in 2D,
    the polynomial tail must have degree m-1 = 2.

    References
    ----------
    Duchon (1977), Splines minimizing rotation-invariant seminorms.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_control_points: Optional[int] = 4000,
    ) -> None:

        self.full_X = X
        self.N_total, self.d = X.shape

        self.X = X
        self.N = self.N_total

        # --------------------------------------------------------------
        # Control point selection (ONLY place where subsampling happens)
        # --------------------------------------------------------------

        if n_control_points is None or n_control_points >= self.N:
            self.control_indices = np.arange(self.N)
            print(f"[Spline] Using all {self.N} points as control points.")
        else:
            print(
                f"[Spline] Selecting {n_control_points} control points "
                f"via k-means (from {self.N} samples) …"
            )
            self.control_indices = self._select_control_points_kmeans(
                self.X, n_control_points
            )

        self.control_points = self.X[self.control_indices]

        # --------------------------------------------------------------
        # Kernel matrix
        # --------------------------------------------------------------

        pairwise_distances = cdist(self.X, self.control_points)
        self.kernel_matrix = self._kernel(pairwise_distances)

        # --------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------

        self.radial_coefficients: Optional[np.ndarray] = None
        self.polynomial_coefficients: Optional[np.ndarray] = None

        self.dof: Optional[float] = None
        self.singular_values: Optional[np.ndarray] = None
        self.lambda_reg: Optional[float] = None


    def _poly_design(self, X: np.ndarray) -> np.ndarray:
        """
        Construct polynomial design matrix up to degree 2.

        Parameters
        ----------
        X : ndarray of shape (N, d)

        Returns
        -------
        P : ndarray of shape (N, p)
            Polynomial feature matrix.
        """
        X = np.asarray(X, float)
        N, d = X.shape

        cols = [np.ones((N, 1))]
        cols.append(X)

        for i in range(d):
            for j in range(i, d):
                cols.append((X[:, i] * X[:, j])[:, None])

        return np.hstack(cols)

    def _poly_dof(self) -> int:
        """
        Number of polynomial basis functions.

        Returns
        -------
        int
            C(d+2, 2)
        """
        return comb(self.d + 2, 2)


    def _kernel(self, r: np.ndarray) -> np.ndarray:
        """
        Polyharmonic kernel φ(r) = r⁴ log r.

        Parameters
        ----------
        r : ndarray
            Pairwise distances.

        Returns
        -------
        ndarray
            Kernel matrix with φ(0) = 0.
        """
        r = np.asarray(r, float)
        result = np.zeros_like(r)
        mask = r > 0
        result[mask] = (r[mask] ** 4) * np.log(r[mask])
        return result


    def _select_control_points_kmeans(self, X, n_control_points):
        """
        Selects control points from the training data using k-means clustering.
        For each cluster center, the training point closest to the center is chosen.

        Args:
            X (np.ndarray): Training input points of shape (N, d).
            n_control_points (int): Number of control points (clusters) desired.

        Returns:
            np.ndarray: Indices of the selected control points.
        """
        kmeans = KMeans(n_clusters=n_control_points, random_state=0)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        
        # For each cluster center, find the index of the closest training point.
        indices = []
        for center in centers:
            distances = np.linalg.norm(X - center, axis=1)
            indices.append(np.argmin(distances))
        return np.array(indices)


    def compute_dof(self, lambda_reg: float) -> float:
        """
        Compute effective degrees of freedom (DoF) of the spline model.

        The radial contribution is computed from the singular values
        of the kernel matrix Φ:

            dof_radial = Σ_i s_i² / (s_i² + λ)

        where s_i are singular values of Φ.

        The polynomial contribution equals the number of polynomial
        basis functions (degree 2):

            dof_poly = C(d+2, 2)

        Parameters
        ----------
        lambda_reg : float
            Ridge regularization parameter λ.

        Returns
        -------
        float
            Effective degrees of freedom.

        Notes
        -----
        This corresponds to the trace of the smoothing matrix
        for ridge-regularized RBF regression.
        """

        if self.singular_values is None:
            _, S, _ = svd(self.kernel_matrix, full_matrices=False)
            self.singular_values = S

        S = self.singular_values
        radial_dof = np.sum(S**2 / (S**2 + lambda_reg))

        poly_dof = self._poly_dof()

        return float(radial_dof + poly_dof)

    
    def find_lambda_for_dof(self, dof, tol=1e-3, fallback_lambda=1e6):
        """
        Finds the regularization parameter (lambda_reg) that achieves a target degrees 
        of freedom using a numerical root solver. Falls back to a fixed lambda if search fails.
        """
        def f(lam):
            return self.compute_dof(lam) - dof

        try:
            f_low = f(1e-8)
            f_high = f(1e8)

            if f_low * f_high > 0:
                raise ValueError(
                    f"The specified dof_target ({dof}) is too small or otherwise infeasible. "
                    f"f(1e-8) = {f_low} and f(1e8) = {f_high} do not have opposite signs."
                )

            sol = root_scalar(f, bracket=[1e-8, 1e8], xtol=tol, method='brentq')
            if sol.converged:
                return sol.root
            else:
                raise RuntimeError("Lambda finding did not converge.")

        except Exception as e:
            print(f"Warning: Lambda search failed: {e}")
            print(f"Setting lambda_reg = {fallback_lambda}")
            computed_dof = self.compute_dof(fallback_lambda)
            print(f"Resulting degrees of freedom: {computed_dof:.2f}")
            return fallback_lambda


    def fit(
        self,
        Y: np.ndarray,
        lambda_reg: Optional[float] = None,
        dof: Optional[float] = None,
    ) -> "Spline":
        """
        Fit spline using ridge regression.

        Parameters
        ----------
        Y : ndarray of shape (N, D)
            Target outputs.

        lambda_reg : float, optional
            Regularization parameter.

        dof : float, optional
            Target effective degrees of freedom.

        Returns
        -------
        self : Spline
        """

        self.weights = None
        self.coeffs = None
        self.dof = None
        self.singular_values = None
        self.lambda_reg = None

        N = self.N
        D = Y.shape[1]
        m = self.control_points.shape[0]

        if dof is not None:
            lambda_reg = self.find_lambda_for_dof(dof)
        if lambda_reg is None:
            lambda_reg = 0.0
        self.lambda_reg = float(lambda_reg)

        kernel_matrix = self.kernel_matrix                      # (N, m)
        P = self._poly_design(self.X)       # (N, p)
        p = P.shape[1]

        X_design = np.hstack((kernel_matrix, P))      # (N, m+p)

        # Ridge block penalizes only radial weights
        reg_block = np.hstack((np.sqrt(self.lambda_reg) * np.eye(m),
                               np.zeros((m, p))))

        A_aug = np.vstack((X_design, reg_block))      # (N+m, m+p)
        Y_aug = np.vstack((Y, np.zeros((m, D))))      # (N+m, D)

        z, *_ = np.linalg.lstsq(A_aug, Y_aug, rcond=None)

        self.radial_coefficients = z[:m]      # (m, D)
        self.polynomial_coefficients = z[m:]  # (p, D)

        self.dof = self.compute_dof(self.lambda_reg)
        return self


    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the fitted spline at new input locations.

        This computes the mapping

            f(x) = Σ_i w_i φ(||x - c_i||) + P₂(x) a

        where:
            - φ(r) = r⁴ log r
            - c_i are control points
            - P₂(x) is the quadratic polynomial basis

        Parameters
        ----------
        X_new : ndarray of shape (M, d) or (d,)
            Evaluation points in input space.

        Returns
        -------
        ndarray of shape (M, D) or (D,)
            Predicted outputs. If a single point is provided,
            a 1D array is returned.

        Notes
        -----
        Internally computes:
            - pairwise distances to control points
            - radial basis evaluation
            - polynomial tail evaluation

        The method assumes the spline has been fitted.
        """
        X_new = np.atleast_2d(X_new).astype(float)
        was_1d = (X_new.shape[0] == 1 and X_new.shape[1] == self.control_points.shape[1])

        pairwise_distances = cdist(X_new, self.control_points, metric="euclidean")
        K_new = self._kernel(pairwise_distances)           # (M, m)

        P_new = self._poly_design(X_new)                      # (M, p)
        predictions = (
            K_new @ self.radial_coefficients
            + P_new @ self.polynomial_coefficients
        )

        return predictions[0] if was_1d else predictions


    def _poly_derivatives(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute first and second derivatives of the quadratic
        polynomial basis P₂(x).

        The polynomial basis contains all monomials up to degree 2:

            [1,
             x_1, ..., x_d,
             x_i x_j for i <= j]

        Parameters
        ----------
        X : ndarray of shape (N, d)
            Evaluation points.

        Returns
        -------
        dPdx : ndarray of shape (N, p, d)
            First derivatives of the polynomial basis.

            dPdx[n, j, i] = ∂P_j / ∂x_i evaluated at X[n].

        d2P : ndarray of shape (p, d, d)
            Second derivatives of the polynomial basis.

            d2P[j, i, k] = ∂²P_j / ∂x_i ∂x_k

            These are constant in x.
        """

        X = np.asarray(X, dtype=float)
        N, d = X.shape

        p = self._poly_dof()

        dPdx = np.zeros((N, p, d), dtype=float)
        d2P = np.zeros((p, d, d), dtype=float)

        idx = 0

        # ---------------------
        # Constant term (1)
        # ---------------------
        idx += 1  # derivative is zero

        # ---------------------
        # Linear terms (x_i)
        # ---------------------
        for i in range(d):
            dPdx[:, idx, i] = 1.0
            idx += 1

        # ---------------------
        # Quadratic terms (x_i x_j, i <= j)
        # ---------------------
        for i in range(d):
            for j in range(i, d):

                if i == j:
                    # x_i^2
                    dPdx[:, idx, i] = 2.0 * X[:, i]
                    d2P[idx, i, i] = 2.0
                else:
                    # x_i x_j
                    dPdx[:, idx, i] = X[:, j]
                    dPdx[:, idx, j] = X[:, i]
                    d2P[idx, i, j] = 1.0
                    d2P[idx, j, i] = 1.0

                idx += 1

        return dPdx, d2P

    
    def compute_jacobians(
        self,
        evaluation_points: np.ndarray,
        eps: float = 1e-12,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """
        Compute Jacobians of the spline mapping at given points.
    
        This evaluates the Jacobian of the polyharmonic spline
        f : ℝᵈ → ℝᴰ using the analytic derivative of the kernel
    
            φ(r) = r⁴ log r
    
        with s = ||x - c||², yielding
    
            ∇φ = (x - c) * s * (2 log s + 1).
    
        Parameters
        ----------
        evaluation_points : ndarray of shape (M, d)
            Points at which to evaluate the Jacobian.
    
        eps : float, default=1e-12
            Small threshold to avoid numerical instability for s ≈ 0.
    
        chunk_size : int or None, optional
            If provided, splits computation into chunks of size ≤ chunk_size
            to reduce memory usage. If None, processes all points at once.
    
        Returns
        -------
        J : ndarray of shape (M, D, d)
            Jacobian matrices evaluated at each point, where
            J[i] = ∂f / ∂x evaluated at evaluation_points[i].
    
        Raises
        ------
        RuntimeError
            If the spline has not been fitted.
    
        Notes
        -----
        The result combines:
            - Radial basis contribution (RBF part)
            - Polynomial contribution (degree ≤ 2)
    
        The implementation uses a BLAS-friendly contraction for efficiency.
        """
    
        if self.radial_coefficients is None:
            raise RuntimeError("Spline must be fitted before evaluation.")
    
        Xev = np.asarray(evaluation_points, float)
    
        centers = self.centers                  # (m, d)
        alpha = self.radial_coefficients        # (m, D)
        beta = self.polynomial_coefficients     # (p, D)
    
        M, d = Xev.shape
        m = centers.shape[0]
        D = alpha.shape[1]
    
        def _compute_block(Xblock):
            Mb = Xblock.shape[0]
    
            diffs = Xblock[:, None, :] - centers[None, :, :]   # (Mb, m, d)
            s = np.sum(diffs * diffs, axis=2)                  # (Mb, m)
    
            factor = np.zeros_like(s)                          # (Mb, m)
            mask = s > eps
            if np.any(mask):
                factor[mask] = s[mask] * (2 * np.log(s[mask]) + 1)
    
            weighted = factor[:, :, None] * diffs              # (Mb, m, d)
    
            weighted_2d = weighted.transpose(1, 0, 2).reshape(m, Mb * d)  # (m, Mb*d)
    
            out = alpha.T @ weighted_2d                        # (D, Mb*d)
    
            return out.reshape(D, Mb, d).transpose(1, 0, 2)    # (Mb, D, d)
    
        # --- RBF contribution ---
        if chunk_size is None:
            J_rbf = _compute_block(Xev)                        # (M, D, d)
        else:
            J_rbf = np.zeros((M, D, d))
            for i in range(0, M, chunk_size):
                sl = slice(i, i + chunk_size)
                J_rbf[sl] = _compute_block(Xev[sl])
    
        # --- polynomial contribution ---
        dPdx, _ = self._poly_derivatives(Xev)                  # (M, p, d)
        J_poly = np.einsum("mpd,pk->mkd", dPdx, beta)          # (M, D, d)
    
        return J_rbf + J_poly    


    def compute_hessians(
        self,
        evaluation_points: np.ndarray,
        eps: float = 1e-12,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """
        Compute Hessians of the spline mapping at given points.
    
        Evaluates the second derivative of the polyharmonic spline
        f : ℝᵈ → ℝᴰ using the analytic Hessian of the kernel
    
            φ(r) = r⁴ log r
    
        with s = ||x - c||². The Hessian decomposes into:
    
            Hφ = 2(2 log s + 3)(x - c)(x - c)ᵀ
                 + s(2 log s + 1) I
    
        Parameters
        ----------
        evaluation_points : ndarray of shape (M, d)
            Points at which to evaluate the Hessian.
    
        eps : float, default=1e-12
            Threshold to avoid numerical instability for s ≈ 0.
    
        chunk_size : int or None, optional
            If provided, splits computation into chunks to reduce memory usage.
    
        Returns
        -------
        H : ndarray of shape (M, D, d, d)
            Hessian matrices evaluated at each point, where
            H[i, k] = ∂²f_k / ∂x∂xᵀ at evaluation_points[i].
    
        Raises
        ------
        RuntimeError
            If the spline has not been fitted.
    
        Notes
        -----
        The result combines:
            - Radial basis contribution (RBF part)
            - Polynomial contribution (constant Hessian)
    
        The implementation uses BLAS-friendly contractions where possible.
        """
    
        if self.radial_coefficients is None:
            raise RuntimeError("Spline must be fitted.")
    
        Xev = np.asarray(evaluation_points, float)
    
        centers = self.centers              # (m, d)
        alpha = self.radial_coefficients    # (m, D)
        beta = self.polynomial_coefficients # (p, D)
    
        M, d = Xev.shape
        m = centers.shape[0]
        D = alpha.shape[1]
    
        I = np.eye(d)                       # (d, d)
    
        def _block(Xblock):
            Mb = Xblock.shape[0]
    
            diffs = Xblock[:, None, :] - centers[None, :, :]   # (Mb, m, d)
            s = np.sum(diffs * diffs, axis=2)                  # (Mb, m)
    
            factor_dd = np.zeros_like(s)                       # (Mb, m)
            factor_I = np.zeros_like(s)                        # (Mb, m)
    
            mask = s > eps
            if np.any(mask):
                log_s = np.log(s[mask])
                factor_dd[mask] = 2.0 * (2.0 * log_s + 3.0)
                factor_I[mask] = s[mask] * (2.0 * log_s + 1.0)
    
            # --- isotropic term (I contribution) ---
            term2 = factor_I @ alpha                           # (Mb, D)
    
            # --- anisotropic term ((x-c)(x-c)^T contribution) ---
            weighted = factor_dd[:, :, None] * diffs           # (Mb, m, d)
    
            weighted_2d = weighted.transpose(1, 0, 2).reshape(m, Mb * d)  # (m, Mb*d)
            tmp = alpha.T @ weighted_2d                        # (D, Mb*d)
            tmp = tmp.reshape(D, Mb, d).transpose(1, 0, 2)     # (Mb, D, d)
    
            # accumulate outer products
            term1 = np.zeros((Mb, D, d, d))                    # (Mb, D, d, d)
    
            for j in range(d):
                weighted_j = weighted[:, :, j]                 # (Mb, m)
                contrib = weighted_j @ alpha                   # (Mb, D)
    
                term1[:, :, j, :] += np.einsum(
                    "mnd,nk,mn->mkd",
                    diffs,
                    alpha,
                    weighted_j,
                )                                              # (Mb, D, d)
    
            H = term1 + term2[:, :, None, None] * I            # (Mb, D, d, d)
            return H
    
        # --- RBF contribution ---
        if chunk_size is None:
            H_rbf = _block(Xev)                                # (M, D, d, d)
        else:
            H_rbf = np.zeros((M, D, d, d))
            for i in range(0, M, chunk_size):
                sl = slice(i, i + chunk_size)
                H_rbf[sl] = _block(Xev[sl])
    
        # --- polynomial contribution (constant) ---
        _, d2P = self._poly_derivatives(Xev)                   # (p, d, d)
        H_poly = np.einsum("pij,pk->kij", d2P, beta)           # (D, d, d)
        H_poly = np.broadcast_to(H_poly[None], (M, D, d, d))   # (M, D, d, d)
    
        return H_rbf + H_poly
    

    def compute_metric(
        self,
        evaluation_points: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute induced Riemannian metric tensor.

        Given Jacobians J(x), the metric is:

            g_ij = ⟨∂f/∂x_i , ∂f/∂x_j⟩

        Parameters
        ----------
        evaluation_points : ndarray of shape (M, d)

        eps : float, optional
            Numerical stability parameter.

        Returns
        -------
        ndarray of shape (M, d, d)
            Metric tensor at each evaluation point.
        """

        jacobians = self.compute_jacobians(evaluation_points, eps=eps)
        metrics = np.einsum('mki,mkj->mij', jacobians, jacobians)
        return metrics


    def compute_christoffels(
        self,
        evaluation_points: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute Christoffel symbols of the induced connection.

        Using:

            Γ^k_{ij} = g^{km} ⟨∂²f/∂x_i∂x_j , ∂f/∂x_m⟩

        Parameters
        ----------
        evaluation_points : ndarray of shape (M, d)

        eps : float, optional
            Numerical stability parameter.

        Returns
        -------
        ndarray of shape (M, d, d, d)
            Christoffel symbols Γ^k_{ij} at each point.

        Notes
        -----
        These define the Levi-Civita connection
        induced by the embedding f.
        """

        J = self.compute_jacobians(evaluation_points, eps=eps)   # (M, out_dim, d)
        H = self.compute_hessians(evaluation_points, eps=eps)    # (M, out_dim, d, d)

        g = np.einsum('mki,mkj->mij', J, J)                      # (M, d, d)  (here d is intrinsic dim)
        g_inv = np.linalg.inv(g)

        S = np.einsum('mkij,mkl->mijl', H, J)                    # (M, d, d, d)
        Gamma = np.einsum('mkm,mijm->mijk', g_inv, S)            # (M, d, d, d)
        return Gamma


    def compute_torsion(
        self,
        evaluation_points: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute torsion tensor.

            T^k_{ij} = Γ^k_{ij} − Γ^k_{ji}

        For a Levi-Civita connection,
        torsion should vanish.

        Parameters
        ----------
        evaluation_points : ndarray of shape (M, d)

        Returns
        -------
        ndarray of shape (M, d, d, d)
            Torsion tensor.
        """

        Gamma = self.compute_christoffels(evaluation_points, eps=eps)
        T = Gamma - np.swapaxes(Gamma, axis1=2, axis2=3)
        return T


    def curvature(
        self,
        evaluation_points: np.ndarray,
        eps: float = 1e-12
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute extrinsic and intrinsic curvature quantities.

        Returns:

        - Second fundamental form projections
        - Riemann curvature tensor
        - Ricci tensor
        - Scalar curvature
        - Normal basis vectors

        Parameters
        ----------
        evaluation_points : ndarray of shape (M, d)

        eps : float, optional
            Numerical stability parameter.

        Returns
        -------
        K : ndarray
            Second fundamental form components.

        Riem : ndarray
            Riemann curvature tensor.

        Ric : ndarray
            Ricci curvature tensor.

        Rsc : ndarray
            Scalar curvature.

        Nmat : ndarray
            Orthonormal normal basis.
        """

        J = self.compute_jacobians(evaluation_points, eps=eps)   # (M, D, 2) in your use
        H = self.compute_hessians(evaluation_points, eps=eps)

        g = np.einsum('mki,mkj->mij', J, J)
        g_inv = np.linalg.inv(g)

        D = J.shape[1]
        U, S, Vh = np.linalg.svd(J, full_matrices=True)
        Nmat = U[:, :, 2:]

        K = np.einsum('mdij,mdA->mAij', H, Nmat)

        h1 = np.einsum('mAik,mAjl->mAijkl', K, K)
        h2 = np.einsum('mAil,mAjk->mAijkl', K, K)
        Riem = np.sum(h1 - h2, axis=1)

        Ric = np.einsum('mik,mijkl->mjl', g_inv, Riem)
        Rsc = np.einsum('mjl,mjl->m', g_inv, Ric)
        return K, Riem, Ric, Rsc, Nmat


    def map_velocities(self, V: np.ndarray) -> np.ndarray:
        """
        Project velocity vectors through the spline tangent.

        Given velocities V in output space, this computes
        their representation in input coordinates using:

            J(x) v_input ≈ v_output

        via least squares.

        Parameters
        ----------
        V : ndarray of shape (N, D)
            Velocity vectors in embedding space.

        Returns
        -------
        ndarray of shape (N, d)
            Mapped velocities in input coordinates.

        Notes
        -----
        This implements a local pseudo-inverse of the Jacobian.
        """

        jacobians = self.compute_jacobians(self.X)
        mapped_velocities = np.zeros_like(self.X)

        for i in range(self.X.shape[0]):
            mapped_velocities[i], _, _, _ = np.linalg.lstsq(jacobians[i], V[i], rcond=None)

        return mapped_velocities

