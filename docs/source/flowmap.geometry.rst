flowmap.geometry Module
=======================

Geometry analysis tools for fitted FlowMap embeddings.

.. rst-class:: api-card

``FixedPointAnalyzer(embedding)``

Create an analyzer for fixed points of a fitted FlowMap velocity field.

**Parameters**

- ``embedding``: fitted ``VectorFieldEmbedder`` with ``X_emb``, ``V_emb``, ``spline``, and ``spline_vf``.

**Output**

Returns a ``FixedPointAnalyzer`` instance.

**Method notes**: :download:`Fixed points and local linearization <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``FixedPointAnalyzer.identify_fixed_points(*, mode="euclidean", grid_resolution=60, speed_smoothing=1.0, speed_quantile_threshold=0.1, max_candidates=20, min_separation=None, refine=True, metric_regularization=1e-8, jacobian_radius=0.15, weighted_jacobian=True, **grid_kwargs)``

Find low-speed candidate locations and classify local dynamics near each candidate.

**Parameters**

- ``mode``: ``"euclidean"`` for embedding-space classification or ``"riemannian"`` for local metric flattening.
- ``grid_resolution``: number of grid points per axis.
- ``speed_smoothing``: Gaussian smoothing strength applied to the grid-speed field.
- ``speed_quantile_threshold``: low-speed quantile used to keep candidate regions.
- ``max_candidates``: maximum number of candidates to refine and classify.
- ``min_separation``: minimum returned fixed-point separation in embedding units.
- ``refine``: locally minimize velocity magnitude from grid candidates.
- ``metric_regularization``: diagonal regularization for Riemannian metric factorization.
- ``jacobian_radius``: compatibility argument for older notebooks.
- ``weighted_jacobian``: compatibility argument for older notebooks.
- ``grid_kwargs``: passed to ``compute_velocity_on_grid``.

**Output**

Returns a list of dictionaries with ``position``, ``jacobian``, ``linearization``, ``eigenvalues``, ``type``, ``speed``, and ``mode``.

**Method notes**: :download:`Fixed points and local linearization <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``LagrangianPathOptimizer(emb, D=1.0, lam=0.0)``

Create an optimizer for least-action transition paths on a fitted FlowMap embedding.

**Parameters**

- ``emb``: fitted ``VectorFieldEmbedder`` with ``X_emb``, ``V_emb``, and ``spline_vf``.
- ``D``: diffusion scale in the action.
- ``lam``: path-length regularization weight.

**Output**

Returns a ``LagrangianPathOptimizer`` instance.

**Method notes**: :download:`Least-action paths <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``LagrangianPathOptimizer.fit_path(start=None, end=None, *, path_init=None, distance_mode="embed", subsample_n=4000, k=20, alpha=0.0, n_segments=100, lr=5e-3, iters=300)``

Compute a transition path between two embedding-space points.

**Parameters**

- ``start``: start coordinate in embedding space.
- ``end``: end coordinate in embedding space.
- ``path_init``: optional initial path; if provided, ``start`` and ``end`` are inferred from it.
- ``distance_mode``: graph initialization space, ``"embed"`` or ``"orig"``.
- ``subsample_n``: number of cells used for graph initialization.
- ``k``: number of nearest neighbors in the graph.
- ``alpha``: velocity-alignment penalty during graph initialization.
- ``n_segments``: number of segments in the resampled path.
- ``lr``: gradient-descent learning rate.
- ``iters``: number of path optimization iterations.

**Output**

Returns a dictionary with ``path_init``, ``path_refined``, and ``dt``.

**Method notes**: :download:`Least-action paths <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``GeneGradientAnalyzer(emb, *, cell_indices=None, max_cells=None, random_state=0, verbose=True)``

Precompute gene-expression Jacobians for path-local or region-local gene-gradient analysis.

**Parameters**

- ``emb``: fitted ``VectorFieldEmbedder`` with gene-level splines.
- ``cell_indices``: optional cells on which to compute Jacobians.
- ``max_cells``: optional random cap when ``cell_indices`` is large.
- ``random_state``: seed for ``max_cells`` subsampling.
- ``verbose``: print progress and tensor shapes.

**Output**

Returns a ``GeneGradientAnalyzer`` instance with ``jacobians``, ``metric``, and ``cell_indices``.

**Method notes**: :download:`Gene gradients <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``GeneGradientAnalyzer.find_neighbors(trajectory, epsilon=0.05)``

Find cells close to a path or trajectory in embedding space.

**Parameters**

- ``trajectory``: ``(n_points, embedding_dim)`` path coordinates.
- ``epsilon``: radius as a fraction of embedding scale.

**Output**

Returns integer cell indices.

**Method notes**: :download:`Gene gradients <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``GeneGradientAnalyzer.compute_relative_gradients(neighbor_indices, gene_indices, weight="magnitude")``

Compute gene-gradient direction and magnitude relative to local velocity.

**Parameters**

- ``neighbor_indices``: cells used for local averaging.
- ``gene_indices``: genes to evaluate.
- ``weight``: averaging mode, ``"magnitude"`` or ``"uniform"``.

**Output**

Returns a dictionary with ``angles``, ``magnitudes``, and ``gene_indices``.

**Method notes**: :download:`Gene gradients <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``compute_flow_curvature(emb, X_emb=None, eps=1e-12)``

Compute velocity, acceleration components, and curvature from fitted FlowMap splines.

**Parameters**

- ``emb``: fitted ``VectorFieldEmbedder``.
- ``X_emb``: optional embedding coordinates at which to evaluate curvature. Defaults to ``emb.X_emb``.
- ``eps``: numerical stabilizer.

**Output**

Returns nested dictionaries for velocity, acceleration, and curvature arrays.

**Method notes**: :download:`Curvature and acceleration decomposition <_static/FlowMap-method.pdf>`.
