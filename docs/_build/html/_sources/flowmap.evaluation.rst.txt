flowmap.evaluation Module
=========================

Evaluation helpers for fitted FlowMap splines.

.. rst-class:: api-card

``SplineFitEvaluator(emb, *, mode="default", spline=None, spline_vf=None, X_ref=None, V_ref=None)``

Create an evaluator for expression and velocity reconstruction quality.

**Parameters**

- ``emb``: fitted ``VectorFieldEmbedder``.
- ``mode``: ``"default"``, ``"gene"``, or ``"custom"``.
- ``spline``: custom expression/reconstruction spline for ``mode="custom"``.
- ``spline_vf``: custom velocity spline for ``mode="custom"``.
- ``X_ref``: reference expression/features for ``mode="custom"``.
- ``V_ref``: reference velocities for ``mode="custom"``.

**Output**

Returns a ``SplineFitEvaluator`` instance.

.. rst-class:: api-card

``SplineFitEvaluator.evaluate(cell_idx=None)``

Evaluate per-feature and mean reconstruction scores.

**Parameters**

- ``cell_idx``: optional cells to evaluate.

**Output**

Returns a dictionary with expression and velocity ``R²`` and correlation scores.
