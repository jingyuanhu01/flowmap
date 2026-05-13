# Cell Cycle Example Data

This directory contains a compact AnnData file for the FlowMap cell-cycle
tutorial:

```python
import anndata as ad

adata = ad.read_h5ad("example_data/cell_cycle/cell_cycle_33genes.h5ad")
X = adata.X
V = adata.layers["velocity"]
phase = adata.obs["cell_cycle_phase"]
phase_continuous = adata.obs["cell_cycle_relative_pos"]
genes = adata.var_names
```

Contents:

- `adata.X`: expression matrix, shape `(2793, 33)`.
- `adata.layers["velocity"]`: velocity matrix, shape `(2793, 33)`.
- `adata.obs["cell_cycle_phase"]`: discrete cell-cycle phase label.
- `adata.obs["cell_cycle_relative_pos"]`: continuous cell-cycle position in `[0, 1]`.
- `adata.obs["cell_cycle_position"]`: continuous cell-cycle position on the original scale.
- `adata.var_names`: selected cell-cycle gene names.

The matrices were extracted from `rpe1_kinetics_processed.h5ad` using
`X_total` for expression and `velocity_T` for velocity. Expression is log1p
transformed and gene-wise centered; velocity is gene-wise standardized.
