flowmap.utils.dataset Module
============================

Small data container used to keep a fitted FlowMap embedder together with the
metadata and matrices needed by tutorials or downstream analysis.

.. rst-class:: api-card

``FlowMapDataset(X, V, embedder=None, obs=None, var=None, layers=None, obsm=None, varm=None, uns=None, metadata=None)``

Store FlowMap matrices and single-cell metadata in one ordinary Python object.
Existing code can keep using ``data.embedder`` exactly like the original
embedder. Printing the object shows only dimensions and available keys, similar
to AnnData.

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

``FlowMapDataset.attach_embedder(embedder)``

Attach or replace the stored embedder after construction.

**Parameters**

- ``embedder``: fitted ``VectorFieldEmbedder`` or compatible object.

**Output**

Returns ``self``.
