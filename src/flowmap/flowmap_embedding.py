"""
flowmap.flowmap_embedding
=========================

High-level geometry-aware embedding pipeline.

This module provides the VectorFieldEmbedder class, which:

1. Computes a geometry-aware distance graph (optional phase distance)
2. Initializes a low-dimensional embedding (UMAP / t-SNE / PCA)
3. Fits polyharmonic splines between embedding and input space
4. Maps vector fields through the learned embedding
5. Optionally refines the embedding via optimization

This is the main user-facing pipeline for FlowMap.
"""

from __future__ import annotations

import random
import warnings
from typing import Optional, Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import umap.umap_ as umap

from .core.phase_distance_solver import PhaseDistanceGraphSolver
from .core.embedding_refiner import (
    EmbeddingSGDRefiner,
    EmbeddingRefiner,
)

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)


# ======================================================================
# VectorFieldEmbedder
# ======================================================================

class VectorFieldEmbedder:
    """
    Geometry-aware embedding for vector field data.

    Parameters
    ----------
    X : ndarray (N, d)
        High-dimensional data.

    V : ndarray (N, d)
        Vector field in input space.

    method : {"umap", "tsne", "pca"}
        Embedding method.

    dist_method : {"phase", "euclidean"}
        Distance computation method.

    dof : float
        Degrees of freedom for geometry spline.

    dof_vf : float
        Degrees of freedom for vector-field spline.

    Notes
    -----
    The pipeline:

        X  --(embedding)-->  X_emb
        X_emb --(Spline)--> X
        X --(pushforward)--> V_emb

    Two splines are fitted:

        1. Geometry spline  : embedding → input space
        2. Velocity spline  : embedding → embedded vector field
    """

    def __init__(
        self,
        X: np.ndarray,
        V: np.ndarray,
        *,
        method: str = "umap",
        spline_type: str = "thin_plate",
        embedding_dim: int = 2,
        dof: float = 30,
        dof_vf: Optional[float] = None,
        dist_method: str = "phase",
        custom_dist: Optional[np.ndarray] = None,
        X_emb: Optional[np.ndarray] = None,
        dist_kwargs: Optional[dict] = None,
        embed_kwargs: Optional[dict] = None,
        spline_kwargs: Optional[dict] = None,
        spline_vf_kwargs: Optional[dict] = None,
        knn_k: int = 30,
        alpha: float = 0.5,
        use_PCA: bool = True,
        pca_components: int = 30,
        n_control_points: int = 4000,
        n_spline_points: Optional[int] = None,
    ) -> None:

        self.X_raw = X.copy()
        self.V_raw = V.copy()

        self.X = X
        self.V = V

        self.method = method.lower()
        self.spline_type = spline_type.lower()
        self.dist_method = dist_method.lower()
        self.embedding_dim = embedding_dim

        self.dof = dof
        self.dof_vf = dof if dof_vf is None else dof_vf

        self.knn_k = knn_k
        self.alpha = alpha
        self.n_control_points = n_control_points
        self.n_spline_points = n_spline_points

        self.custom_dist = custom_dist
        self.embed_kwargs = embed_kwargs or {}
        self.dist_kwargs = dist_kwargs or {}
        self.spline_kwargs = spline_kwargs or {}
        self.spline_vf_kwargs = spline_vf_kwargs or {}

        self.spline: Optional[Any] = None
        self.spline_vf: Optional[Any] = None

        # Optional PCA pre-projection
        if use_PCA and X.shape[1] > 50:
            self._apply_pca(pca_components)

        self.X_emb = X_emb


    def _apply_pca(self, n_components: int) -> None:
        print(f"[PCA] Projecting to {n_components} components …")
        svd = TruncatedSVD(n_components=n_components, random_state=0)

        self.X = svd.fit_transform(self.X)
        self.V = self.V @ svd.components_.T
        self._pca = svd


    def _compute_distance_graph(self):

        if self.custom_dist is not None:
            return self.custom_dist

        if self.dist_method == "phase":
            solver = PhaseDistanceGraphSolver(
                self.X,
                self.V,
                k=self.knn_k,
                alpha=self.alpha,
                **self.dist_kwargs,
            )
            return solver.compute_graph()

        return None


    def _subsample(self, X, V, X_emb, seed=1):
        if self.n_spline_points is None:
            return X, V, X_emb

        n = min(self.n_spline_points, X.shape[0])
        np.random.seed(seed)
        idx = np.random.choice(X.shape[0], n, replace=False)

        return X[idx], V[idx], X_emb[idx]


    def fit_embedding(
        self,
        seed: Optional[int] = None,
    ) -> None:
        """
        Compute the initial embedding and fit geometry-aware splines.

        This method performs:

        1. Distance graph computation (if using phase distance)
        2. Low-dimensional embedding (e.g. UMAP / t-SNE / PCA)
        3. Spline fitting:
            - spline : Spline
            - spline_vf : Spline

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Notes
        -----
        After calling this method, the following attributes are available:

        - X_emb : ndarray (N, embedding_dim)
            Embedded coordinates.
        - V_emb : ndarray (N, embedding_dim)
            Embedded vector field.
        - spline : Spline
        - spline_vf : Spline

        This method must be called before `refine_embedding()`.
        """


        np.random.seed(seed)
        random.seed(seed)

        if self.X_emb is None:

            if self.dist_method == "phase":
                print("[FlowMap] Computing phase distance graph …")
                self.dist_graph = self._compute_distance_graph()

                if self.method == "umap":
                    reducer = umap.UMAP(
                        metric="precomputed",
                        random_state=seed,
                        n_components=self.embedding_dim,
                        **self.embed_kwargs,
                    )
                elif self.method == "tsne":
                    reducer = TSNE(
                        metric="precomputed",
                        init="random",
                        random_state=seed,
                        n_components=self.embedding_dim,
                        **self.embed_kwargs,
                    )
                else:
                    raise ValueError("Unsupported embedding method.")

                self.X_emb = reducer.fit_transform(self.dist_graph)

            else:
                if self.method == "umap":
                    reducer = umap.UMAP(
                        metric="euclidean",
                        random_state=seed,
                        n_components=self.embedding_dim,
                        **self.embed_kwargs,
                    )
                    self.X_emb = reducer.fit_transform(self.X)
                elif self.method == "tsne":
                    reducer = TSNE(
                        metric="euclidean",
                        init="random",
                        random_state=seed,
                        n_components=self.embedding_dim,
                        **self.embed_kwargs,
                    )
                    self.X_emb = reducer.fit_transform(self.X)
                elif self.method == "pca":
                    svd = TruncatedSVD(
                        n_components=self.embedding_dim,
                        random_state=seed,
                    )
                    self.X_emb = svd.fit_transform(self.X)
                else:
                    raise ValueError("Unsupported embedding method.")

        self.X_emb_init = self.X_emb.copy()

        self._fit_splines(self.X_emb)

        print("[FlowMap] Done.")


    def _get_spline_class(self):
        if self.spline_type == "polyharmonic":
            from .core.polyharmonic_spline import PolyharmonicSpline
            return PolyharmonicSpline
        elif self.spline_type == "thin_plate":
            from .core.thin_plate_spline import ThinPlateSpline
            return ThinPlateSpline
        else:
            raise ValueError(
                f"Unknown spline_type: {self.spline_type}. "
                "Use 'polyharmonic' or 'thin_plate'."
            )


    def _fit_splines(self, X_emb: np.ndarray) -> None:
        SplineClass = self._get_spline_class()

        # Optional subsampling
        if self.n_spline_points is not None:
            X_fit, V_fit, X_emb_fit = self._subsample(self.X, self.V, X_emb)
        else:
            X_fit, V_fit, X_emb_fit = self.X, self.V, X_emb

        print(f"[Spline] Fitting manifold spline on {X_fit.shape[0]} cells …")

        self.spline = SplineClass(
            X_emb_fit,
            n_control_points=self.n_control_points,
            **self.spline_kwargs,
        )
        self.spline.fit(X_fit, dof=self.dof)

        print("[Spline] Mapping vector field …")
        vec_field = self.spline.map_velocities(V_fit)

        print("[Spline] Fitting velocity spline …")

        self.spline_vf = SplineClass(
            X_emb_fit,
            n_control_points=self.n_control_points,
            **self.spline_vf_kwargs,
        )
        self.spline_vf.fit(vec_field, dof=self.dof_vf)

        self.V_emb = self.spline_vf.predict(self.X_emb)


    def fit_gene_level_splines(
        self,
        dof_gene: float = 80,
        dof_vf_gene: float = 80,
        X: Optional[np.ndarray] = None,
        V: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit gene-level splines for expression and velocity fields.
        """

        SplineClass = self._get_spline_class()
        if self.X_emb is None:
            raise RuntimeError("Run `fit_embedding()` before fitting gene splines.")

        # --------------------------------------------------------------
        # Determine data source
        # --------------------------------------------------------------
        if X is not None and V is not None:
            X_gene, V_gene = X, V
        elif hasattr(self, "X_raw") and hasattr(self, "V_raw"):
            X_gene, V_gene = self.X_raw, self.V_raw
        else:
            raise RuntimeError("No gene-level data available.")

        self.X_gene = X_gene
        self.V_gene = V_gene

        # Optional subsampling
        if self.n_spline_points is not None:
            X_fit, V_fit, X_emb_fit = self._subsample(X_gene, V_gene, self.X_emb)
        else:
            X_fit, V_fit, X_emb_fit = X_gene, V_gene, self.X_emb

        # --------------------------------------------------------------
        # Fit gene-expression spline
        # --------------------------------------------------------------
        print(f"[Spline-Gene] Fitting expression spline on {X_fit.shape[0]} cells …")

        self.spline_gene = SplineClass(
            X_emb_fit,
            n_control_points=self.n_control_points,
            **self.spline_kwargs,
        )
        self.spline_gene.fit(X_fit, dof=dof_gene)

        # --------------------------------------------------------------
        # Map gene velocities through expression spline
        # --------------------------------------------------------------
        print("[Spline-Gene] Mapping velocities …")

        V_emb_gene = self.spline_gene.map_velocities(V_fit)

        # --------------------------------------------------------------
        # Fit gene-velocity spline
        # --------------------------------------------------------------
        print("[Spline-Gene] Fitting velocity spline …")

        self.spline_vf_gene = SplineClass(
            X_emb_fit,
            n_control_points=self.n_control_points,
            **self.spline_vf_kwargs,
        )
        self.spline_vf_gene.fit(V_emb_gene, dof=dof_vf_gene)

        print(
            f"[Spline-Gene] Done. "
            f"X_gene {self.X_gene.shape}, "
            f"V_gene {self.V_gene.shape}"
        )

    
    def refine_embedding(
        self,
        lam: Optional[float] = None,
        method: str = "batch",
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 50,
        tol: float = 1e-2,
        verbose: bool = False,
    ) -> None:
        """
        Refine the embedding using geometry-aware optimization.

        This optional step adjusts embedding coordinates to improve
        geometric consistency between:

            - Manifold reconstruction
            - Vector field pushforward

        Parameters
        ----------
        lam : float, optional
            Velocity reconstruction weight.

        method : {"batch", "global"}, default="batch"
            Optimization strategy:
                - "batch"  : mini-batch SGD
                - "global" : full-batch L-BFGS

        batch_size : int, default=64
            Mini-batch size (batch mode).

        lr : float, default=1e-3
            Learning rate (batch mode).

        epochs : int, default=50
            Number of epochs (batch mode).

        tol : float, default=1e-2
            Tolerance (global mode).

        verbose : bool, default=False
            Whether to print optimization progress.

        Notes
        -----
        This method:

        1. Updates ``self.X_emb``
        2. Re-fits splines
        3. Updates ``self.V_emb``
        """

        if self.spline is None:
            self._fit_splines(self.X_emb)

        # --------------------------------------------------------------
        # Batch mode (SGD)
        # --------------------------------------------------------------
        if method == "batch":

            refiner = EmbeddingSGDRefiner(
                emb=self,
                lam=lam,
            )

            self.X_emb = refiner.refine(
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                verbose=verbose,
            )

        # --------------------------------------------------------------
        # Global mode (L-BFGS)
        # --------------------------------------------------------------
        elif method == "global":

            refiner = EmbeddingRefiner(
                emb=self,
                lam=lam,
            )

            self.X_emb, self.opt_result = refiner.refine(
                tol=tol,
                verbose=verbose,
            )

        else:
            raise ValueError("Unknown optimisation method.")

        # --------------------------------------------------------------
        # Re-fit splines after refinement
        # --------------------------------------------------------------
        print("[FlowMap] Re-fitting splines after refinement …")
        self._fit_splines(self.X_emb)

