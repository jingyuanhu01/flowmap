# flowmap/geometry/gene_gradient_analyzer.py

from __future__ import annotations
import numpy as np
from typing import Optional, Dict


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

    def __init__(
        self,
        emb,
        *,
        cell_indices: Optional[np.ndarray] = None,
        max_cells: Optional[int] = None,
        random_state: Optional[int] = 0,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        emb : VectorFieldEmbedder
            Must contain:

            - X_emb
            - V_emb
            - spline_gene
            - spline_vf_gene
        cell_indices : ndarray, optional
            Cells on which to precompute gene Jacobians and metrics. Indices
            refer to rows of ``emb.X_emb``.
        max_cells : int, optional
            If provided and ``cell_indices`` contains more cells than this,
            randomly subset to ``max_cells`` before computing Jacobians.
        random_state : int, optional
            Seed used when ``max_cells`` triggers random subsampling.
        verbose : bool, default=True
            Print progress messages for expensive Jacobian computations.
        """

        if not hasattr(emb, "spline_gene"):
            raise RuntimeError("Gene-level splines not fitted.")

        self.emb = emb
        n_cells = emb.X_emb.shape[0]
        n_genes = getattr(emb, "X_gene", np.empty((n_cells, 0))).shape[1]

        if cell_indices is None:
            cell_indices = np.arange(n_cells)
        else:
            cell_indices = np.asarray(cell_indices, dtype=int)

        if max_cells is not None and len(cell_indices) > max_cells:
            rng = np.random.default_rng(random_state)
            cell_indices = np.sort(
                rng.choice(cell_indices, size=max_cells, replace=False)
            )

        self.cell_indices = np.asarray(cell_indices, dtype=int)
        self._cell_lookup = {idx: i for i, idx in enumerate(self.cell_indices)}

        if verbose:
            print(
                "[GeneGradient] Computing expression Jacobians for "
                f"{len(self.cell_indices)} cells x {n_genes} genes …"
            )
            if len(self.cell_indices) < n_cells:
                print(
                    "[GeneGradient] Using a cell subset "
                    f"({len(self.cell_indices)} of {n_cells} cells)."
                )

        X_eval = emb.X_emb[self.cell_indices]
        self.J = emb.spline_gene.compute_jacobians(X_eval)
        self.jacobians = self.J
        self.metric = emb.spline_gene.compute_metric(X_eval)

        if verbose:
            print(
                "[GeneGradient] Done. "
                f"J {self.J.shape}, metric {self.metric.shape}"
            )


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
            - gene_indices : ndarray
                Gene indices corresponding to the returned angles and
                magnitudes. This may be shorter than the input if no valid
                local velocity was available.
        """

        angles = []
        magnitudes = []

        local_indices = [
            self._cell_lookup[int(idx)]
            for idx in np.asarray(neighbor_indices, dtype=int)
            if int(idx) in self._cell_lookup
        ]

        if len(local_indices) == 0:
            return dict(
                angles=np.asarray(angles),
                magnitudes=np.asarray(magnitudes),
            )

        gene_indices = np.asarray(gene_indices, dtype=int)
        J = self.J[local_indices][:, gene_indices, :]
        metric = self.metric[local_indices]
        velocity = self.emb.V_emb[self.cell_indices[local_indices]]

        grad = np.linalg.solve(metric, np.swapaxes(J, 1, 2))
        grad = np.swapaxes(grad, 1, 2)

        L = np.linalg.cholesky(
            metric + 1e-8 * np.eye(metric.shape[-1])[None, :, :]
        )
        grad_E = np.einsum("cij,cgj->cgi", L, grad)
        vel_E = np.einsum("cij,cj->ci", L, velocity)

        v_norm = np.linalg.norm(vel_E, axis=1)
        valid_cells = v_norm > 1e-8
        if not np.any(valid_cells):
            return dict(
                angles=np.asarray(angles),
                magnitudes=np.asarray(magnitudes),
            )

        grad_E = grad_E[valid_cells]
        vel_E = vel_E[valid_cells]
        v_norm = v_norm[valid_cells]

        e1 = vel_E / v_norm[:, None]
        if e1.shape[1] != 2:
            raise ValueError("GeneGradientAnalyzer currently expects 2D embeddings.")
        e2 = np.column_stack([-e1[:, 1], e1[:, 0]])

        comp_parallel = np.einsum("cgi,ci->cg", grad_E, e1)
        comp_orth = np.einsum("cgi,ci->cg", grad_E, e2)

        if weight == "magnitude":
            weights = np.linalg.norm(grad_E, axis=2)
        elif weight == "uniform":
            weights = np.ones_like(comp_parallel)
        else:
            raise ValueError("weight must be 'magnitude' or 'uniform'.")

        vec = np.stack([comp_parallel, comp_orth], axis=2)
        total_w = weights.sum(axis=0)
        valid_genes = total_w > 0
        mean_vec = np.zeros((len(gene_indices), 2), dtype=float)
        mean_vec[valid_genes] = (
            (weights[:, valid_genes, None] * vec[:, valid_genes]).sum(axis=0)
            / total_w[valid_genes, None]
        )

        angles = np.arctan2(mean_vec[valid_genes, 1], mean_vec[valid_genes, 0])
        magnitudes = np.linalg.norm(mean_vec[valid_genes], axis=1)

        return dict(
            angles=np.asarray(angles),
            magnitudes=np.asarray(magnitudes),
            gene_indices=gene_indices[valid_genes],
        )
