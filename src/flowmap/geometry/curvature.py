"""
Curvature analysis for FlowMap embeddings.

Compute geometric quantities for trajectories induced by the embedded
vector field on the reconstructed expression manifold.

The trajectory in gene space is

    γ(t) = ψ(z(t))

where ψ is the spline manifold and z(t) follows the embedded velocity
field. Acceleration decomposes into:

    A = A_along + A_steer + A_normal

and curvature satisfies:

    k_total² = k_geod² + k_normal²
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
            total   : full ambient acceleration
            along   : acceleration along trajectory
            steer   : steering acceleration (within manifold)
            normal  : acceleration orthogonal to manifold

        curvature
            total   : total curvature
            geodesic : turning within manifold
            normal  : extrinsic curvature
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
    A_normal = A - A_tan

    a_along = np.einsum("nd,nd->n", A_tan, T)

    A_along = a_along[:, None] * T
    A_steer = A_tan - A_along

    # --------------------------------------------------
    # Curvature
    # --------------------------------------------------
    k_geod = np.linalg.norm(A_steer, axis=1) / (speed2 + eps)
    k_normal = np.linalg.norm(A_normal, axis=1) / (speed2 + eps)

    k_total = np.sqrt(k_geod**2 + k_normal**2)

    # --------------------------------------------------
    # Structured output
    # --------------------------------------------------
    return {
        "velocity": {
            "total": V,
        },
        "acceleration": {
            "total": A,
            "along": A_along,
            "steer": A_steer,
            "normal": A_normal,
        },
        "curvature": {
            "total": k_total,
            "geodesic": k_geod,
            "normal": k_normal,
        },
    }
