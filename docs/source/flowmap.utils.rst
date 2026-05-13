flowmap.utils Module
====================

Utility functions used by plotting and geometry analysis.

.. rst-class:: api-card

``compute_velocity_on_grid(X_emb, spline_vf=None, *, grid_size=100, grid_density=1.0, smooth=0.5, n_neighbors=None, min_mass=0.01, margin_ratio=0.0, return_mesh=False)``

Evaluate a fitted velocity field on a masked 2D grid.

**Parameters**

- ``X_emb``: embedding coordinates.
- ``spline_vf``: fitted velocity spline with ``predict``.
- ``grid_size``: grid resolution per axis.
- ``grid_density``: density scaling for grid construction.
- ``smooth``: density smoothing used for the mask.
- ``n_neighbors``: optional neighborhood count for density estimation.
- ``min_mass``: low-density grid mask threshold.
- ``margin_ratio``: padding around embedding bounds.
- ``return_mesh``: include mesh arrays for plotting.

**Output**

Returns grid coordinates, a mask, grid velocities, and optionally the mesh.

.. rst-class:: api-card

``FlowMapDataset(X, V, embedder=None, obs=None, var=None, layers=None, obsm=None, varm=None, uns=None, metadata=None)``

Store FlowMap matrices and single-cell metadata in one ordinary Python object. Existing code can keep using ``data.embedder`` exactly like the original embedder.

**Parameters**

- ``X``: ``(n_cells, n_features)`` data matrix used by FlowMap.
- ``V``: ``(n_cells, n_features)`` velocity matrix aligned to ``X``.
- ``embedder``: optional fitted ``VectorFieldEmbedder`` or compatible object.
- ``obs``: optional cell metadata table.
- ``var``: optional gene or feature metadata table.
- ``layers``: optional named matrices, usually copied from AnnData layers.
- ``obsm``: optional cell-level embeddings or matrices.
- ``varm``: optional feature-level matrices.
- ``uns``: optional unstructured metadata.
- ``metadata``: optional FlowMap-specific metadata.

**Output**

Returns a ``FlowMapDataset`` instance.

.. rst-class:: api-card

``from_embedder(embedder, adata=None, copy_layers=None, copy_obsm=None, copy_varm=None, copy_uns=False, metadata=None)``

Wrap an existing fitted embedder in a ``FlowMapDataset``. If ``adata`` is provided, selected single-cell metadata are copied into the wrapper.

**Parameters**

- ``embedder``: fitted ``VectorFieldEmbedder`` or compatible object with ``X`` and ``V``.
- ``adata``: optional AnnData object used as the metadata source.
- ``copy_layers``: layer names to copy from ``adata.layers``; use ``True`` to copy all.
- ``copy_obsm``: keys to copy from ``adata.obsm``; use ``True`` to copy all.
- ``copy_varm``: keys to copy from ``adata.varm``; use ``True`` to copy all.
- ``copy_uns``: copy ``adata.uns`` when ``True``.
- ``metadata``: optional FlowMap-specific metadata.

**Output**

Returns a ``FlowMapDataset`` with the original object available as ``data.embedder``.

.. rst-class:: api-card

``from_anndata(adata, X_layer=None, V_layer="velocity", X_obsm=None, V_obsm=None, X_emb_obsm=None, V_emb_obsm=None, embedder=None, copy_layers=None, copy_obsm=None, copy_varm=None, copy_uns=False, metadata=None)``

Create a ``FlowMapDataset`` directly from AnnData matrices and metadata.

**Parameters**

- ``adata``: AnnData-like object.
- ``X_layer``: optional layer key for the data matrix.
- ``V_layer``: layer key for the velocity matrix. Defaults to ``"velocity"``.
- ``X_obsm``: optional ``obsm`` key for the data matrix.
- ``V_obsm``: optional ``obsm`` key for the velocity matrix.
- ``X_emb_obsm``: optional ``obsm`` key copied as ``"X_flowmap"``.
- ``V_emb_obsm``: optional ``obsm`` key copied as ``"V_flowmap"``.
- ``embedder``: optional fitted FlowMap embedder to store on the dataset.
- ``copy_layers``: layer names to copy; use ``True`` to copy all.
- ``copy_obsm``: ``obsm`` keys to copy; use ``True`` to copy all.
- ``copy_varm``: ``varm`` keys to copy; use ``True`` to copy all.
- ``copy_uns``: copy ``adata.uns`` when ``True``.
- ``metadata``: optional FlowMap-specific metadata.

**Output**

Returns a ``FlowMapDataset``.

.. rst-class:: api-card

``to_anndata(data)``

Convert a ``FlowMapDataset`` back to an AnnData object.

**Parameters**

- ``data``: ``FlowMapDataset`` instance.

**Output**

Returns an AnnData object. If ``data.X`` is reduced feature space but ``data.var`` describes genes, the original ``var`` table is stored under ``adata.uns["flowmap"]``.

.. rst-class:: api-card

``save_dataset(data, path, **joblib_kwargs)``

Save a ``FlowMapDataset`` as a regular ``.joblib`` file.

**Parameters**

- ``data``: ``FlowMapDataset`` instance.
- ``path``: output path.
- ``joblib_kwargs``: additional keyword arguments passed to ``joblib.dump``.

**Output**

Returns ``None``.

.. rst-class:: api-card

``load_dataset(path)``

Load a ``FlowMapDataset`` saved with ``save_dataset``.

**Parameters**

- ``path``: input ``.joblib`` path.

**Output**

Returns a ``FlowMapDataset``.
