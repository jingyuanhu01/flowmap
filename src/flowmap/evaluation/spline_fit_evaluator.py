# flowmap/evaluation/spline_fit_evaluator.py

from __future__ import annotations
import numpy as np
from typing import Optional, Dict


class SplineFitEvaluator:
    """
    Evaluate reconstruction accuracy of manifold and velocity splines.

    This class quantifies how well a fitted spline model reconstructs:

        1. Feature values (e.g. genes or principal components)
        2. Vector field values

    It supports evaluation at two levels:

        - Gene-level splines (``spline_gene``, ``spline_vf_gene``)
        - Embedding-level splines (``spline``, ``spline_vf``)

    If ``mode`` is not specified, gene-level splines are used when
    available; otherwise embedding-level splines are evaluated.

    Notes
    -----
    Let :math:`f(z)` denote the spline mapping from embedding
    coordinates :math:`z` to feature space, and let
    :math:`J(z) = \\nabla f(z)` denote its Jacobian.

    Expression reconstruction is evaluated via:

    .. math::

        R^2 = 1 - \\frac{\\|X - \\hat{X}\\|^2}{\\|X - \\bar{X}\\|^2}

    Velocity reconstruction is evaluated using the local linear
    pushforward:

    .. math::

        \\hat{V} = J(z) \\beta,

    where :math:`\\beta` is obtained by least-squares fitting to
    the observed velocity.

    This provides both global and per-feature metrics.
    """

    def __init__(
        self,
        emb,
        *,
        mode: str = "default",
        spline=None,
        spline_vf=None,
        X_ref: Optional[np.ndarray] = None,
        V_ref: Optional[np.ndarray] = None,
    ):
        """
        Initialize spline reconstruction evaluator.

        Parameters
        ----------
        emb : VectorFieldEmbedder
            Fitted embedding object.

        mode : {"default", "gene", "custom"}
            - "default": use emb.spline / emb.spline_vf
            - "gene":    use emb.spline_gene / emb.spline_vf_gene
            - "custom":  use user-provided spline / spline_vf

        spline : optional
            Custom spline (used if mode="custom")

        spline_vf : optional
            Custom velocity spline (used if mode="custom")

        X_ref, V_ref : optional
            Custom reference data (required for mode="custom")

        cell_idx : optional
            Subset of cells to evaluate.
        """

        self.emb = emb

        if mode not in {"default", "gene", "custom"}:
            raise ValueError("mode must be 'default', 'gene', or 'custom'.")

        self.mode = mode

        # --------------------------------------------------------------
        # Select splines + reference data
        # --------------------------------------------------------------

        if mode == "default":
            self.spline = emb.spline
            self.spline_vf = emb.spline_vf
            self.X_ref = emb.X
            self.V_ref = emb.V

        elif mode == "gene":
            if not hasattr(emb, "spline_gene"):
                raise RuntimeError("Gene-level splines not fitted.")
            self.spline = emb.spline_gene
            self.spline_vf = emb.spline_vf_gene
            self.X_ref = emb.X_gene
            self.V_ref = emb.V_gene

        elif mode == "custom":
            if spline is None:
                raise ValueError("Custom mode requires 'spline'.")
            if spline_vf is None:
                raise ValueError("Custom mode requires 'spline_vf'.")
            if X_ref is None or V_ref is None:
                raise ValueError("Custom mode requires X_ref and V_ref.")

            self.spline = spline
            self.spline_vf = spline_vf
            self.X_ref = X_ref
            self.V_ref = V_ref

        self.X_emb = emb.X_emb
        self.V_emb = emb.V_emb

        self.X_pred_all = self.spline.predict(self.X_emb)
        self.J_all = self.spline.compute_jacobians(self.X_emb)


    def evaluate(self, cell_idx: Optional[np.ndarray] = None) -> Dict:
        """
        Compute reconstruction metrics.

        Returns
        -------
        dict
            Dictionary containing:

            expr_r2 : float
                Mean expression R² across features.

            vel_r2 : float
                Mean velocity R² across features.

            expr_r2_gene : list of float
                Per-feature expression R² values.

            vel_r2_gene : list of float
                Per-feature velocity R² values.

            expr_corr_gene : list of float
                Per-feature Pearson correlation (expression).

            vel_corr_gene : list of float
                Per-feature Pearson correlation (velocity).

            X_pred : ndarray
                Reconstructed feature values.

            V_pred : ndarray
                Reconstructed velocity values.

        Notes
        -----
        Expression reconstruction evaluates the spline prediction:

        .. math::

            \\hat{X} = f(z)

        Velocity reconstruction evaluates the local linear pushforward:

        .. math::

            \\hat{V} = J(z) \\beta

        where :math:`\\beta` is obtained via least-squares fit at
        each cell independently.
        """

        if cell_idx is None:
            idx = np.arange(self.X_emb.shape[0])
        else:
            idx = np.asarray(cell_idx)

        X_obs = self.X_ref[idx]
        V_obs = self.V_ref[idx]
        X_pred = self.X_pred_all[idx]
        J = self.J_all[idx]

        N, G = X_obs.shape

        # --- reconstruct velocity via Jacobian pushforward ---
        V_pred = np.zeros_like(V_obs)

        for i in range(N):
            beta, *_ = np.linalg.lstsq(J[i], V_obs[i], rcond=None)
            V_pred[i] = J[i] @ beta

        # --- R² computation ---
        def r2(A, B):
            SSE = np.sum((A - B) ** 2, axis=0)
            TSS = np.sum((A - A.mean(axis=0)) ** 2, axis=0) + 1e-12
            return 1.0 - SSE / TSS

        expr_r2_gene = r2(X_obs, X_pred)
        vel_r2_gene = r2(V_obs, V_pred)

        expr_r2 = float(np.mean(expr_r2_gene))
        vel_r2 = float(np.mean(vel_r2_gene))

        # --- correlations ---
        def corr_per_feature(A, B):
            out = []
            for i in range(A.shape[1]):
                if np.std(A[:, i]) > 1e-12 and np.std(B[:, i]) > 1e-12:
                    out.append(float(np.corrcoef(A[:, i], B[:, i])[0, 1]))
                else:
                    out.append(np.nan)
            return np.array(out)

        expr_corr = corr_per_feature(X_obs, X_pred)
        vel_corr = corr_per_feature(V_obs, V_pred)

        return dict(
            expr_r2=expr_r2,
            vel_r2=vel_r2,
            expr_r2_gene=expr_r2_gene.tolist(),
            vel_r2_gene=vel_r2_gene.tolist(),
            expr_corr_gene=expr_corr.tolist(),
            vel_corr_gene=vel_corr.tolist(),
            X_pred=X_pred,
            V_pred=V_pred,
        )
