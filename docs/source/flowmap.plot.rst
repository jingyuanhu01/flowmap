flowmap.plot Module
===================

Small plotting helpers for fitted embeddings and vector fields.

.. rst-class:: api-card

``plot_velocity_stream(X_2d, spline=None, V=None, scatter_color=None, grid_size=50, grid_density=1.0, stream_density=1.0, title=None, scatter_size=10, scatter_alpha=0.5, arrowsize=1.5, ax=None, figsize=(8, 6), aspect="equal", cmap="tab10", vmin=None, vmax=None, show_axes=False, show_colorbar=False, streamline_thickness=4.0, pad_frac=0.0)``

Plot a 2D velocity field as streamlines over an embedding.

**Parameters**

- ``X_2d``: embedding coordinates.
- ``spline``: fitted velocity spline.
- ``V``: raw velocity vectors used only when ``spline`` is not provided.
- ``scatter_color``: optional point color values or labels.
- ``grid_size``: streamplot grid resolution.
- ``stream_density``: streamline density.
- ``ax``: optional Matplotlib axis.

**Output**

Returns a Matplotlib figure.

.. rst-class:: api-card

``plot_velocity_grid(X_emb, spline_vf=None, V=None, grid_size=50, grid_density=1.0, smooth=0.5, n_neighbors=None, min_mass=0.01, scatter_color=None, scatter_size=80, scatter_alpha=0.1, arrow_color="black", arrow_alpha=0.9, arrow_scale=3.0, arrow_width=0.0025, figsize=(6, 6), cmap="viridis", vmin=None, vmax=None, show_axes=False, show_colorbar=False)``

Plot a 2D velocity field as grid arrows.

**Parameters**

- ``X_emb``: embedding coordinates.
- ``spline_vf``: fitted velocity spline.
- ``V``: raw velocity vectors used only when ``spline_vf`` is not provided.
- ``grid_size``: grid resolution.
- ``scatter_color``: optional point color values or labels.
- ``arrow_scale``: quiver scale.

**Output**

Returns a Matplotlib figure.

.. rst-class:: api-card

``plot_2d_quiver(X, V, color=None, scale=1.0, normalize=False, size=10, alpha=0.5, cmap="coolwarm", title="")``

Plot points and 2D vectors using Matplotlib quiver.

**Parameters**

- ``X``: point coordinates.
- ``V``: vectors.
- ``color``: optional numeric point/vector color.
- ``normalize``: normalize vectors before plotting.

**Output**

Returns a Matplotlib figure.

.. rst-class:: api-card

``plot_3d_quiver(points, derivatives, points_color=None, s=20, alpha=0.3, arrow_size=0.2, normalize=True, title="", cmap="coolwarm", show_colorbar=True, show_axes=True)``

Plot points and 3D vectors using Matplotlib quiver.

**Parameters**

- ``points``: 3D point coordinates.
- ``derivatives``: 3D vectors.
- ``points_color``: optional numeric point/vector color.

**Output**

Returns a Matplotlib figure.

.. rst-class:: api-card

``plot_3d_scatter(points, points_color=None, title="", azim=-60, elev=9, grid_interval=2, dot_size=5, alpha=0.4)``

Plot a 3D scatter view.

**Parameters**

- ``points``: 3D point coordinates.
- ``points_color``: optional numeric point color.
- ``azim`` and ``elev``: view angles.

**Output**

Returns a Matplotlib figure.
