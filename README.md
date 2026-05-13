# FlowMap

FlowMap is a Python package for geometry-aware vector field embeddings, with a
focus on RNA velocity and other dynamical single-cell data. It embeds
high-dimensional observations together with their velocities, reconstructs a
smooth low-dimensional vector field, and provides tools for downstream
geometric analysis.

## Installation

FlowMap can be installed directly from GitHub:

```bash
pip install git+https://github.com/jingyuanhu01/flowmap.git
```

### Minimal Dependencies

FlowMap is built on standard scientific Python packages, including NumPy,
SciPy, scikit-learn, UMAP, statsmodels, and matplotlib. These are installed
automatically with FlowMap.

You do not need a special single-cell environment to use the core API.

### Optional Single-Cell Environment

For RNA velocity workflows, it is often useful to work in a fresh environment
with common single-cell tools installed.

Using conda:

```bash
conda create -n flowmap python=3.11
conda activate flowmap
```

Using mamba:

```bash
mamba create -n flowmap python=3.11
mamba activate flowmap
```

Then install FlowMap and optional RNA velocity tools:

```bash
pip install git+https://github.com/jingyuanhu01/flowmap.git
pip install scanpy anndata scvelo
```

Scanpy, AnnData, and scVelo are not required by the core FlowMap API, but they
are commonly used for loading, preprocessing, and exploring single-cell RNA
velocity data.

### Development Install

If you want to edit the source code locally:

```bash
git clone https://github.com/jingyuanhu01/flowmap.git
cd flowmap
pip install -e .
```
