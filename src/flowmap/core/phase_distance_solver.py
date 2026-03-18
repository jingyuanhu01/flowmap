"""
flowmap.phase_distance_solver
=============================

Phase-aware distance graph construction for geometry-aware embeddings.

This module implements the hybrid phase distance used in FlowMap:

    d_ij = sqrt( ||x_i - x_j||² + λ (t_i - t_j)² )

where t_i are least-squares time estimates derived from local velocity
consistency constraints.

The resulting graph is returned as a symmetric sparse COO matrix and
can be used with UMAP or t-SNE (metric="precomputed").
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from typing import Optional


# ======================================================================
# PhaseDistanceGraphSolver
# ======================================================================

class PhaseDistanceGraphSolver:
    """
    Construct hybrid phase-aware distance graph.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Input positions.

    V : ndarray of shape (n, d)
        Velocity vectors in input space.

    k : int, default=30
        Number of nearest neighbors for graph construction.

    alpha : float, default=0.5
        Weight controlling relative contribution of temporal term.

    Notes
    -----
    The method:

    1. Build kNN graph in Euclidean space.
    2. Estimate pairwise least-squares time parameters (t_ls).
    3. Construct hybrid distance:

           d² = ||x_i - x_j||² + λ (t_i - t_j)²

       where λ is automatically scaled to balance magnitudes.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        X: np.ndarray,
        V: np.ndarray,
        k: int = 30,
        alpha: float = 0.5,
    ) -> None:

        self.X = np.asarray(X, float)
        self.V = np.asarray(V, float)

        self.k = int(k)
        self.alpha = float(alpha)

        self.n, self.d = self.X.shape

        # Build kNN graph
        self._build_knn_graph()

        # Precompute least-squares time estimates
        self.t_ls = self._compute_tls()

    # ------------------------------------------------------------------
    # kNN graph
    # ------------------------------------------------------------------

    def _build_knn_graph(self) -> None:
        """
        Construct directed kNN graph (without self-loops).
        """

        nn = NearestNeighbors(
            n_neighbors=self.k + 1,
            algorithm="auto"
        ).fit(self.X)

        distances, neighbors = nn.kneighbors(self.X)

        # Remove self-neighbor
        neighbors = neighbors[:, 1:]

        self.i_idx = np.repeat(np.arange(self.n), self.k)
        self.j_idx = neighbors.flatten()
        self.n_edges = self.n * self.k

    # ------------------------------------------------------------------
    # Least-squares time estimation
    # ------------------------------------------------------------------

    def _compute_tls(self) -> np.ndarray:
        """
        Compute pairwise least-squares time parameters.

        For each edge (i, j), solve:

            min_{t_i, t_j} || x_i + t_i v_i - x_j - t_j v_j ||²

        Returns
        -------
        t_ls : ndarray of shape (n_edges, 2)
            Estimated (t_i, t_j) for each edge.
        """

        X = self.X
        V = self.V
        i_idx, j_idx = self.i_idx, self.j_idx

        # Precompute norms
        v_norm_sq = np.einsum("nd,nd->n", V, V)
        v_dot = np.einsum("nd,nd->n", V[i_idx], V[j_idx])

        # Construct 2x2 system matrix Q per edge
        q00 = v_norm_sq[i_idx]
        q11 = v_norm_sq[j_idx]
        q01 = -v_dot
        q10 = -v_dot

        # Determinant with small stabilizer
        det = q00 * q11 - q01 * q10
        det += 1e-12

        # Inverse of Q
        inv_Q = np.empty((self.n_edges, 2, 2), dtype=float)
        inv_Q[:, 0, 0] = q11 / det
        inv_Q[:, 1, 1] = q00 / det
        inv_Q[:, 0, 1] = -q01 / det
        inv_Q[:, 1, 0] = -q10 / det

        # Construct RHS vector B
        v_dot_x = np.einsum("nd,nd->n", V, X)
        vx_ij = np.einsum("nd,nd->n", V[i_idx], X[j_idx])
        vx_ji = np.einsum("nd,nd->n", V[j_idx], X[i_idx])

        B = np.zeros((self.n_edges, 2), dtype=float)
        B[:, 0] = vx_ij - v_dot_x[i_idx]
        B[:, 1] = vx_ji - v_dot_x[j_idx]

        # Solve
        t_ls = np.einsum("eij,ej->ei", inv_Q, B)

        return t_ls

    # ------------------------------------------------------------------
    # Hybrid graph construction
    # ------------------------------------------------------------------

    def compute_graph(self) -> coo_matrix:
        """
        Construct symmetric hybrid distance graph.

        Returns
        -------
        graph : scipy.sparse.coo_matrix, shape (n, n)
            Symmetric hybrid distance matrix.
        """

        i_idx = self.i_idx
        j_idx = self.j_idx
        X = self.X
        t_ls = self.t_ls

        # Spatial distances
        dist_sq = np.sum((X[i_idx] - X[j_idx]) ** 2, axis=1)

        # Temporal distances
        t_diff_sq = (t_ls[:, 0] - t_ls[:, 1]) ** 2

        # Balance scaling
        norm_spatial = np.linalg.norm(dist_sq)
        norm_temporal = np.linalg.norm(t_diff_sq) + 1e-12

        lam = self.alpha * norm_spatial / norm_temporal

        hybrid_dist = np.sqrt(dist_sq + lam * t_diff_sq)

        # Symmetrize
        i_sym = np.concatenate([i_idx, j_idx])
        j_sym = np.concatenate([j_idx, i_idx])
        d_sym = np.concatenate([hybrid_dist, hybrid_dist])

        graph = coo_matrix(
            (d_sym, (i_sym, j_sym)),
            shape=(self.n, self.n),
        )

        return graph
