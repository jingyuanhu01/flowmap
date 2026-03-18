from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import svd
from scipy.optimize import root_scalar
from sklearn.cluster import KMeans
from math import comb
from typing import Optional, Tuple


class ThinPlateSpline:
    """
    Thin Plate Spline (TPS) regression model.

    Legacy implementation used for reproducing previous results.

    Notes
    -----
    - Uses dimension-aware TPS kernel
    - Polynomial tail is affine (degree 1)
    - Jacobians and Hessians follow original formulation
    - Interface aligned with `Spline` for compatibility
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
        Construct affine polynomial design matrix (degree 1).
    
        Parameters
        ----------
        X : ndarray of shape (N, d)
    
        Returns
        -------
        P : ndarray of shape (N, d+1)
            Polynomial feature matrix [1, x_1, ..., x_d].
        """
        X = np.asarray(X, float)
        N, d = X.shape
    
        return np.hstack((np.ones((N, 1)), X))

    def _poly_dof(self) -> int:
        """
        Number of polynomial basis functions (affine).
    
        Returns
        -------
        int
            d + 1
        """
        return self.d + 1


    def _kernel(self, r: np.ndarray) -> np.ndarray:
        """
        Thin Plate Spline kernel (dimension-aware).
    
        φ_d(r) =
            r^2 log r    if d == 2
            r            if d == 3
            r^(2 - d)    otherwise (up to scaling)
    
        Ensures φ(0) = 0.
        """
        r = np.asarray(r, float)
        result = np.zeros_like(r)
    
        mask = r > 0
    
        if self.d == 2:
            result[mask] = (r[mask] ** 2) * np.log(r[mask])
        elif self.d == 3:
            result[mask] = r[mask]
        else:
            result[mask] = r[mask] ** (2 - self.d)
    
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
        Evaluate the fitted Thin Plate Spline at new input locations.
        
        This computes the mapping
        
            f(x) = Σ_i w_i φ(||x - c_i||) + P₁(x) a
        
        where:
            - φ(r) is the TPS kernel (dimension-dependent)
            - c_i are control points
            - P₁(x) = [1, x_1, ..., x_d] is the affine polynomial basis
        
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
            - affine polynomial evaluation
        
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


    def compute_jacobians(self, evaluation_points, eps=1e-10):
        """
        Compute the Jacobians of the Thin Plate Spline (TPS) function for multiple evaluation points.

        Parameters:
            evaluation_points (np.ndarray): Evaluation points of shape (num_evals, d) in d-dimensional space.
        
        Uses:
            self.control_points (np.ndarray): Control points (num_controls, d).
            self.weights (np.ndarray): TPS weights of shape (num_controls, target_dim).
            self.coeffs (np.ndarray): Affine transformation coefficients of shape (d+1, target_dim).

        Returns:
            np.ndarray: Jacobians for all evaluation points with shape (num_evals, target_dim, d).
        """
        control_points = self.control_points
        weights = self.weights
        coeffs = self.coeffs

        # Get dimensions.
        num_evals, d = evaluation_points.shape  # Number of evaluation points & space dimension.
        num_controls = control_points.shape[0]    # Number of control points.
        target_dim = weights.shape[1]             # Output dimension.

        # Compute pairwise differences: shape (num_evals, num_controls, d)
        diffs = evaluation_points[:, None, :] - control_points[None, :, :]

        # Compute distances r_i(x_j): shape (num_evals, num_controls, 1)
        r_i = np.linalg.norm(diffs, axis=2, keepdims=True)

        # Mask to avoid division by zero.
        valid_mask = r_i > 1e-10

        # Compute normalized direction vectors (only where r_i > 0).
        direction_vectors = np.zeros_like(diffs)
        direction_vectors[valid_mask[..., 0], :] = diffs[valid_mask[..., 0], :] / r_i[valid_mask[..., 0]]

        # Compute the derivative term.
        derivative_term = np.zeros_like(r_i)
        derivative_term[valid_mask] = (1 + 2 * np.log(r_i[valid_mask])) * r_i[valid_mask]

        # Compute derivative contributions (fully vectorized).
        derivative_vectors = weights.T @ (derivative_term * direction_vectors)
        # derivative_vectors now has shape (target_dim, num_evals, d)

        # Add linear (affine) part.
        affine_term = coeffs[1:, :].T[None, :, :]  # shape (1, target_dim, d)

        # Combine derivative and affine terms.
        Jacobians = derivative_vectors + affine_term  # shape (target_dim, num_evals, d)

        return Jacobians

    
    def compute_hessians(self, evaluation_points, eps=1e-10):
        # Extract TPS parameters
        control_points = self.control_points
        weights = self.weights

        # Get dimensions
        M, d_dim = evaluation_points.shape  # Number of evaluation points and input dimension
        N = control_points.shape[0]         # Number of control points
        output_dim = weights.shape[1]       # Number of output dimensions

        # Compute differences for all eval points and control points: shape (M, N, d)
        d = evaluation_points[:, None, :] - control_points[None, :, :]

        # Compute r = ||d|| for each pair (M, N)
        r = np.linalg.norm(d, axis=2) + eps  # Shape (M, N)
        r = r[:, :, None, None]  # Reshape for broadcasting: (M, N, 1, 1)

        # Identity matrix (d, d)
        I = np.eye(d_dim)

        # Compute the outer product efficiently for all eval points: shape (M, N, d, d)
        outer = np.einsum('mni,mnj->mnij', d, d)

        # Compute Hessian for all control points at once: shape (M, N, d, d)
        H_individual = 2 * outer / (r**2) + (2 * np.log(r) + 1) * I

        # Compute weighted sum over control points: shape (M, output_dim, d, d)
        H = np.einsum("mnij,nk->mkij", H_individual, weights)

        return H
    

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

