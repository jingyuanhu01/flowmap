# flowmap/geometry/gene_gradient_analyzer.py

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Tuple


class GeneGradientAnalyzer:
    """
    Analyze gene-level gradients in embedding space.

    This class evaluates the geometry of gene-expression gradients
    reconstructed via the gene-level spline.

    For each cell, the spline Jacobian provides:

        J(z) = ∂X_gene / ∂z

    Combined with the pullback metric, this allows computing:

        - Riemannian gradients
        - Alignment with local velocity
        - Parallel / orthogonal components

    No plotting is performed here. Visualization should be handled
    in the plotting module.
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, emb):
        """
        Parameters
        ----------
        emb : VectorFieldEmbedder
            Must contain:

            - X_emb
            - V_emb
            - spline_gene
            - spline_vf_gene
        """

        if not hasattr(emb, "spline_gene"):
            raise RuntimeError("Gene-level splines not fitted.")

        self.emb = emb

        # Precompute Jacobians and pullback metric
        self.J = emb.spline_gene.compute_jacobians(emb.X_emb)
        self.metric = emb.spline_gene.compute_metric(emb.X_emb)


    # ------------------------------------------------------------------
    # Neighborhood selection
    # ------------------------------------------------------------------

    def find_neighbors(
        self,
        trajectory: np.ndarray,
        epsilon: float = 0.05,
    ) -> np.ndarray:
        """
        Find cells near a trajectory in embedding space.

        Parameters
        ----------
        trajectory : ndarray (m, d)
            Path in embedding space.
        epsilon : float, default=0.05
            Relative neighborhood radius (fraction of embedding scale).

        Returns
        -------
        ndarray
            Indices of neighboring cells.
        """

        X = self.emb.X_emb
        scale = np.mean(np.ptp(X, axis=0))
        radius = epsilon * scale

        distances = np.linalg.norm(
            X[:, None, :] - trajectory[None, :, :],
            axis=2
        ).min(axis=1)

        return np.where(distances <= radius)[0]


    # ------------------------------------------------------------------
    # Gradient decomposition
    # ------------------------------------------------------------------

    def compute_relative_gradients(
        self,
        neighbor_indices: np.ndarray,
        gene_indices: np.ndarray,
        weight: str = "magnitude",
    ) -> Dict[str, np.ndarray]:
        """
        Compute relative orientation of gene gradients to velocity.

        Parameters
        ----------
        neighbor_indices : ndarray
            Cells near region of interest.
        gene_indices : ndarray
            Genes to analyze.
        weight : {"magnitude", "uniform"}, default="magnitude"
            Weighting scheme for averaging across cells.

        Returns
        -------
        dict
            Contains:

            - angles : ndarray
                Mean angle (radians) relative to velocity direction.
            - magnitudes : ndarray
                Mean projected gradient magnitude.
        """

        angles = []
        magnitudes = []

        for g_idx in gene_indices:

            vec_sum = np.zeros(2)
            total_w = 0.0

            for c_idx in neighbor_indices:

                g = self.metric[c_idx]
                J_cell = self.J[c_idx]
                v = self.emb.V_emb[c_idx]

                # Riemannian gradient
                grad = np.linalg.solve(g, J_cell[g_idx])

                # Local orthonormal frame
                L = np.linalg.cholesky(g + 1e-8 * np.eye(g.shape[0]))

                grad_E = L @ grad
                vel_E = L @ v

                v_norm = np.linalg.norm(vel_E)
                if v_norm < 1e-8:
                    continue

                e1 = vel_E / v_norm
                e2 = np.array([-e1[1], e1[0]])

                comp_parallel = grad_E @ e1
                comp_orth = grad_E @ e2

                w = (
                    np.linalg.norm(grad_E)
                    if weight == "magnitude"
                    else 1.0
                )

                vec_sum += w * np.array([comp_parallel, comp_orth])
                total_w += w

            if total_w > 0:
                mean_vec = vec_sum / total_w
                angles.append(np.arctan2(mean_vec[1], mean_vec[0]))
                magnitudes.append(np.linalg.norm(mean_vec))

        return dict(
            angles=np.asarray(angles),
            magnitudes=np.asarray(magnitudes),
        )

