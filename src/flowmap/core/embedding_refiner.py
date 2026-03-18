"""
flowmap.core.embedding_refiner
==============================

Geometry-aware embedding refinement.

This module provides two refinement strategies:

1. EmbeddingRefiner      : Full-batch L-BFGS optimization
2. EmbeddingSGDRefiner   : Mini-batch stochastic refinement

Both operate directly on a fitted ``VectorFieldEmbedder`` object.

The optimized objective is:

    L = ||X - spline(X_emb)||²
        + λ ||V - J_spline(X_emb) · spline_vf(X_emb)||²
        + α ||X_emb - X_emb_init||²

This improves:

- Manifold reconstruction
- Velocity pushforward consistency
- Embedding smoothness
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from tqdm import trange


# ======================================================================
# Full-batch refinement (L-BFGS)
# ======================================================================

class EmbeddingRefiner:
    """
    Global embedding refinement using full-batch optimization.

    Parameters
    ----------
    emb : VectorFieldEmbedder
        Fitted embedding object.
        Must expose:
            - X
            - V
            - X_emb
            - spline
            - spline_vf

    lam : float, optional
        Weight for velocity reconstruction term.
        If None, estimated automatically.

    alpha : float, default=0.0
        Anchor regularization weight.
    """

    def __init__(self, emb, lam: float | None = None, alpha: float = 0.0):

        self.emb = emb

        self.X = emb.X
        self.V = emb.V
        self.X_emb = emb.X_emb

        self.spline = emb.spline
        self.spline_vf = emb.spline_vf

        self.X_init = self.X_emb.copy()

        self.lam = lam if lam is not None else self._estimate_lambda()
        self.alpha = alpha


    # ------------------------------------------------------------------
    # Lambda estimation
    # ------------------------------------------------------------------

    def _estimate_lambda(self, batch_size=100, ratio=10.0):
        idx = np.random.choice(len(self.X), size=min(batch_size, len(self.X)), replace=False)

        X_batch = self.X[idx]
        V_batch = self.V[idx]
        X_emb_batch = self.X_emb[idx]

        X_pred = self.spline.predict(X_emb_batch)
        loss1 = np.sum((X_batch - X_pred) ** 2)

        vf_pred = self.spline_vf.predict(X_emb_batch)
        J = self.spline.compute_jacobians(X_emb_batch)
        V_pred = np.einsum("naj,nj->na", J, vf_pred)
        loss2 = np.sum((V_batch - V_pred) ** 2)

        if loss2 < 1e-8:
            return 1.0

        return np.clip((loss1 / loss2) * ratio, 1e-3, 1e3)


    # ------------------------------------------------------------------
    # Loss + gradient
    # ------------------------------------------------------------------

    def _loss_and_grad(self, X_flat):

        X_emb = X_flat.reshape(self.X_emb.shape)

        # --- manifold reconstruction ---
        X_pred = self.spline.predict(X_emb)
        J = self.spline.compute_jacobians(X_emb)
        H = self.spline.compute_hessians(X_emb)

        loss1 = np.sum((self.X - X_pred) ** 2)

        # --- velocity reconstruction ---
        vf_pred = self.spline_vf.predict(X_emb)
        V_pred = np.einsum("naj,nj->na", J, vf_pred)
        loss2 = np.sum((self.V - V_pred) ** 2)

        # --- anchor ---
        anchor_diff = X_emb - self.X_init
        anchor_loss = np.sum(anchor_diff ** 2)

        # --- gradients ---
        grad1 = -2 * np.einsum("nai,na->ni", J, (self.X - X_pred))

        A = -2 * (self.V - V_pred)
        B = np.einsum("naij,nj->nai", H, vf_pred)
        C = np.einsum("naj,nji->nai", J, self.spline_vf.compute_jacobians(X_emb))
        grad2 = np.einsum("na,nai->ni", A, (B + C))

        grad_anchor = 2 * anchor_diff

        total_loss = loss1 + self.lam * loss2 + self.alpha * anchor_loss
        total_grad = grad1 + self.lam * grad2 + self.alpha * grad_anchor

        return total_loss, total_grad.flatten()


    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def refine(self, method="L-BFGS-B", tol=1e-4, verbose=False):
        """
        Run global embedding optimization.

        Returns
        -------
        X_optimized : ndarray (N, d)
        result : OptimizeResult
        """

        X_flat = self.X_emb.flatten()

        result = minimize(
            fun=lambda x: self._loss_and_grad(x)[0],
            x0=X_flat,
            jac=lambda x: self._loss_and_grad(x)[1],
            method=method,
            tol=tol,
            options={"gtol": tol, "ftol": tol, "disp": verbose},
        )

        X_new = result.x.reshape(self.X_emb.shape)

        # Update embedding object
        self.emb.X_emb = X_new

        return X_new, result


# ======================================================================
# Mini-batch refinement (SGD)
# ======================================================================

class EmbeddingSGDRefiner:
    """
    Mini-batch stochastic embedding refinement.

    Parameters
    ----------
    emb : VectorFieldEmbedder
        Fitted embedding object.

    lam : float, optional
        Velocity reconstruction weight.

    alpha : float, default=1.0
        Anchor regularization weight.
    """

    def __init__(self, emb, lam: float | None = None, alpha: float = 1.0):

        self.emb = emb

        self.X = emb.X
        self.V = emb.V
        self.X_emb = emb.X_emb

        self.spline = emb.spline
        self.spline_vf = emb.spline_vf

        self.X_init = self.X_emb.copy()

        self.lam = lam if lam is not None else 1.0
        self.alpha = alpha


    def _batch_loss_grad(self, X_flat, batch_idx):

        X_emb = X_flat.reshape(self.X_emb.shape)

        X_batch = X_emb[batch_idx]
        X_ref = self.X[batch_idx]
        V_ref = self.V[batch_idx]
        X_init = self.X_init[batch_idx]

        X_pred = self.spline.predict(X_batch)
        J = self.spline.compute_jacobians(X_batch)
        H = self.spline.compute_hessians(X_batch)

        vf_pred = self.spline_vf.predict(X_batch)
        V_pred = np.einsum("naj,nj->na", J, vf_pred)

        grad1 = -2 * np.einsum("nai,na->ni", J, (X_ref - X_pred))

        A = -2 * (V_ref - V_pred)
        B = np.einsum("naij,nj->nai", H, vf_pred)
        C = np.einsum("naj,nji->nai", J, self.spline_vf.compute_jacobians(X_batch))
        grad2 = np.einsum("na,nai->ni", A, (B + C))

        grad_anchor = 2 * (X_batch - X_init)

        total_grad = np.zeros_like(X_emb)
        total_grad[batch_idx] = grad1 + self.lam * grad2 + self.alpha * grad_anchor

        return total_grad.flatten()


    def refine(self, epochs=50, batch_size=64, lr=1e-2, seed=1, verbose=False):
        """
        Run mini-batch embedding refinement.

        Returns
        -------
        X_optimized : ndarray (N, d)
        """

        rng = np.random.default_rng(seed)

        X_flat = self.X_emb.flatten()
        N = self.X_emb.shape[0]
        indices = np.arange(N)

        for _ in trange(epochs, desc="Embedding refinement"):
            rng.shuffle(indices)

            for i in range(0, N, batch_size):
                batch_idx = indices[i:i + batch_size]
                grad = self._batch_loss_grad(X_flat, batch_idx)
                X_flat -= lr * np.nan_to_num(grad)

        X_new = X_flat.reshape(self.X_emb.shape)

        # Update embedding object
        self.emb.X_emb = X_new

        return X_new

