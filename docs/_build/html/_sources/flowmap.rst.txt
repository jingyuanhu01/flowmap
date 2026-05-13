FlowMap Module
==============

High-level entry point for fitting FlowMap embeddings and splines.

.. rst-class:: api-card

``VectorFieldEmbedder(X, V, *, method="umap", spline_type="thin_plate", embedding_dim=2, dof=30, dof_vf=None, dist_method="phase", custom_dist=None, X_emb=None, dist_kwargs=None, embed_kwargs=None, spline_kwargs=None, spline_vf_kwargs=None, knn_k=30, alpha=0.5, use_PCA=True, pca_components=30, n_control_points=4000, n_spline_points=None)``

Create a FlowMap embedding object from observations and velocities. Call ``fit_embedding()`` to compute the embedding and fit the main splines.

**Parameters**

- ``X``: ``(n_cells, n_features)`` data matrix.
- ``V``: ``(n_cells, n_features)`` velocity matrix aligned to ``X``.
- ``method``: embedding backend, one of ``"umap"``, ``"tsne"``, or ``"pca"``.
- ``spline_type``: spline backend, ``"thin_plate"`` or ``"polyharmonic"``.
- ``embedding_dim``: output embedding dimension. Most plotting and geometry helpers expect ``2``.
- ``dof``: effective degrees of freedom for the reconstruction spline.
- ``dof_vf``: effective degrees of freedom for the velocity-field spline. Defaults to ``dof``.
- ``dist_method``: distance mode, ``"phase"`` or ``"euclidean"``.
- ``custom_dist``: precomputed distance graph or matrix used instead of FlowMap distance construction.
- ``X_emb``: existing embedding coordinates. If provided, FlowMap skips embedding initialization and fits splines on these coordinates.
- ``dist_kwargs``: keyword arguments passed to the distance solver.
- ``embed_kwargs``: keyword arguments passed to UMAP or t-SNE.
- ``spline_kwargs``: keyword arguments passed to the reconstruction spline.
- ``spline_vf_kwargs``: keyword arguments passed to the velocity spline.
- ``knn_k``: number of neighbors for phase-distance graph construction.
- ``alpha``: phase-distance velocity/time weighting parameter.
- ``use_PCA``: project high-dimensional input before fitting when ``X`` has many features.
- ``pca_components``: number of PCA/SVD components when ``use_PCA=True``.
- ``n_control_points``: number of spline control points.
- ``n_spline_points``: optional random subset size for spline fitting.

**Output**

Returns a ``VectorFieldEmbedder`` instance. After fitting, commonly used attributes are ``X_emb``, ``V_emb``, ``spline``, and ``spline_vf``.

**Method notes**: :download:`FlowMap embedding and spline fitting <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``VectorFieldEmbedder.fit_embedding(seed=None)``

Compute the low-dimensional embedding and fit the reconstruction and velocity-field splines.

**Parameters**

- ``seed``: optional random seed for embedding initialization and reproducibility.

**Output**

Returns ``None``. Updates ``X_emb``, ``V_emb``, ``spline``, and ``spline_vf`` on the object.

**Method notes**: :download:`FlowMap embedding <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``VectorFieldEmbedder.fit_gene_level_splines(dof_gene=80, dof_vf_gene=80, X=None, V=None)``

Fit splines from embedding coordinates to gene expression and gene velocity. Use this before gene-level evaluation or gradient analysis.

**Parameters**

- ``dof_gene``: effective degrees of freedom for the gene-expression spline.
- ``dof_vf_gene``: effective degrees of freedom for the gene-velocity spline.
- ``X``: optional gene-expression matrix. If omitted, FlowMap uses stored raw input when available.
- ``V``: optional gene-velocity matrix aligned to ``X``.

**Output**

Returns ``None``. Adds ``X_gene``, ``V_gene``, ``spline_gene``, and ``spline_vf_gene`` to the object.

**Method notes**: :download:`Gene-level reconstruction <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``VectorFieldEmbedder.refine_embedding(lam=None, method="batch", batch_size=64, lr=1e-3, epochs=50, tol=1e-2, verbose=False)``

Optionally refine embedding coordinates after the initial fit. Most tutorials do not need this step.

**Parameters**

- ``lam``: velocity reconstruction weight.
- ``method``: optimization mode, ``"batch"`` or ``"global"``.
- ``batch_size``: mini-batch size for ``method="batch"``.
- ``lr``: learning rate for ``method="batch"``.
- ``epochs``: number of optimization epochs for ``method="batch"``.
- ``tol``: convergence tolerance for ``method="global"``.
- ``verbose``: print optimization progress.

**Output**

Returns ``None``. Updates ``X_emb`` and refits ``spline`` and ``spline_vf``.

**Method notes**: :download:`Embedding refinement <_static/FlowMap-method.pdf>`.
