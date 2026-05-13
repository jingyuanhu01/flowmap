flowmap.core Module
===================

Lower-level spline, distance, and refinement utilities used by the high-level FlowMap pipeline. Most users call these indirectly through ``VectorFieldEmbedder``.

.. rst-class:: api-card

``ThinPlateSpline(X, n_control_points=4000)``

Create a thin-plate spline model with optional control-point selection.

**Parameters**

- ``X``: input/control coordinates, usually embedding coordinates.
- ``n_control_points``: maximum number of control points; ``None`` uses all points.

**Output**

Returns a spline object.

**Method notes**: :download:`Spline reconstruction <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``ThinPlateSpline.fit(Y, lambda_reg=None, dof=None)``

Fit spline targets at the constructor coordinates.

**Parameters**

- ``Y``: target matrix.
- ``lambda_reg``: ridge regularization value.
- ``dof``: target effective degrees of freedom; FlowMap chooses ``lambda_reg`` from this when provided.

**Output**

Returns ``self``.

**Method notes**: :download:`Spline reconstruction <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``ThinPlateSpline.predict(X_new)``

Evaluate fitted spline values.

**Parameters**

- ``X_new``: evaluation coordinates.

**Output**

Returns predicted target values.

**Method notes**: :download:`Spline reconstruction <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``ThinPlateSpline.compute_jacobians(evaluation_points, eps=1e-10)``

Evaluate spline Jacobians at coordinates.

**Parameters**

- ``evaluation_points``: coordinates at which to compute Jacobians.
- ``eps``: numerical stabilizer.

**Output**

Returns an array of Jacobian matrices.

**Method notes**: :download:`Pullback metric and gene gradients <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``ThinPlateSpline.compute_metric(evaluation_points, eps=1e-8)``

Compute pullback metric tensors from spline Jacobians.

**Parameters**

- ``evaluation_points``: coordinates at which to compute metric tensors.
- ``eps``: numerical stabilizer.

**Output**

Returns metric tensors.

**Method notes**: :download:`Riemannian metric <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``ThinPlateSpline.map_velocities(V)``

Map velocities through the fitted spline geometry.

**Parameters**

- ``V``: velocities in target/input coordinates.

**Output**

Returns velocities represented in spline input coordinates.

**Method notes**: :download:`Velocity pushforward/pullback <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``PolyharmonicSpline(X, n_control_points=4000)``

Create a polyharmonic spline model. The public methods mirror ``ThinPlateSpline``.

**Parameters**

- ``X``: input/control coordinates.
- ``n_control_points``: maximum number of control points; ``None`` uses all points.

**Output**

Returns a spline object.

**Method notes**: :download:`Spline reconstruction <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``PhaseDistanceGraphSolver(X, V, k=30, alpha=0.5)``

Create a solver for FlowMap's phase-aware neighbor graph.

**Parameters**

- ``X``: input data matrix.
- ``V``: velocity matrix aligned to ``X``.
- ``k``: nearest neighbors per cell.
- ``alpha``: temporal/velocity contribution weight.

**Output**

Returns a ``PhaseDistanceGraphSolver`` instance.

**Method notes**: :download:`Phase distance graph <_static/FlowMap-method.pdf>`.

.. rst-class:: api-card

``PhaseDistanceGraphSolver.compute_graph()``

Build the symmetric sparse phase-distance graph.

**Parameters**

- None.

**Output**

Returns a ``scipy.sparse.coo_matrix`` distance graph.

**Method notes**: :download:`Phase distance graph <_static/FlowMap-method.pdf>`.
