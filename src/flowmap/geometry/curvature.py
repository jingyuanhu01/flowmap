"""
Flow geometry analysis for FlowMap embeddings.

Compute geometric quantities for trajectories induced by the embedded
vector field on the reconstructed expression manifold.

The trajectory in gene space is

    γ(t) = ψ(z(t))

where ψ is the spline manifold and z(t) follows the embedded velocity
field. Acceleration decomposes into:

    A = A_flow + A_steer + A_surface

where

    flow    : acceleration along the trajectory
    steer   : turning acceleration within the manifold
    surface : acceleration normal to the manifold

Curvature satisfies:

    k_total² = k_steer² + k_surface²
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict


def compute_flow_curvature(
    emb,
    X_emb: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute velocity, acceleration decomposition, and curvature.

    Parameters
    ----------
    emb : VectorFieldEmbedder
        Must expose:

        - ``spline``     : manifold spline ψ
        - ``spline_vf``  : embedded vector field
        - ``X_emb``      : embedding coordinates

    X_emb : ndarray (N, d), optional
        Evaluation points. Defaults to ``emb.X_emb``.

    eps : float
        Numerical stability constant.

    Returns
    -------
    dict
        Structured dictionary:

        velocity
            total : ambient velocity

        acceleration
            total    : full ambient acceleration
            flow     : acceleration along trajectory
            steer    : turning acceleration within manifold
            surface  : acceleration orthogonal to manifold

        curvature
            total    : total curvature
            steer    : intrinsic turning curvature
            surface  : extrinsic curvature
    """

    if X_emb is None:
        X_emb = emb.X_emb

    spline = emb.spline
    spline_vf = emb.spline_vf

    # --------------------------------------------------
    # Geometry of ψ
    # --------------------------------------------------
    J = spline.compute_jacobians(X_emb)       # (N, D, d)
    H = spline.compute_hessians(X_emb)        # (N, D, d, d)

    # --------------------------------------------------
    # Embedded vector field
    # --------------------------------------------------
    v = spline_vf.predict(X_emb)              # (N, d)
    Jv = spline_vf.compute_jacobians(X_emb)   # (N, d, d)

    # --------------------------------------------------
    # Ambient velocity
    # --------------------------------------------------
    V = np.einsum("ndk,nk->nd", J, v)

    speed2 = np.einsum("nd,nd->n", V, V)
    speed = np.sqrt(speed2 + eps)

    T = V / speed[:, None]

    # --------------------------------------------------
    # Ambient acceleration
    # --------------------------------------------------
    A1 = np.einsum("ndij,ni,nj->nd", H, v, v)

    a_u = np.einsum("nij,nj->ni", Jv, v)

    A2 = np.einsum("ndk,nk->nd", J, a_u)

    A = A1 + A2

    # --------------------------------------------------
    # Tangent / normal projectors
    # --------------------------------------------------
    g = np.einsum("ndk,ndl->nkl", J, J)
    g_inv = np.linalg.inv(g + eps * np.eye(g.shape[-1]))

    # --------------------------------------------------
    # Acceleration decomposition
    # --------------------------------------------------
    JTA = np.einsum("ndk,nd->nk", J, A)              # J^T A, shape (N, d)
    coef = np.einsum("nkl,nl->nk", g_inv, JTA)       # (J^T J)^(-1) J^T A
    A_tan = np.einsum("ndk,nk->nd", J, coef)         # J coef

    a_flow = np.einsum("nd,nd->n", A_tan, T)

    A_flow = a_flow[:, None] * T
    A_steer = A_tan - A_flow
    A_surface = A - A_tan

    # --------------------------------------------------
    # Curvature
    # --------------------------------------------------
    k_steer = np.linalg.norm(A_steer, axis=1) / (speed2 + eps)
    k_surface = np.linalg.norm(A_surface, axis=1) / (speed2 + eps)

    k_total = np.sqrt(k_steer**2 + k_surface**2)

    # --------------------------------------------------
    # Structured output
    # --------------------------------------------------
    return {
        "velocity": {
            "total": V,
        },
        "acceleration": {
            "total": A,
            "flow": A_flow,
            "steer": A_steer,
            "surface": A_surface,
        },
        "curvature": {
            "total": k_total,
            "steer": k_steer,
            "surface": k_surface,
        },
    }
