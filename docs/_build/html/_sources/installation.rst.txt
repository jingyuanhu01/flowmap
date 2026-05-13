Installation
============

FlowMap can be installed directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/jingyuanhu01/flowmap.git

Minimal Dependencies
--------------------

One advantage of FlowMap is that the core package uses standard scientific
Python tools. The main dependencies are NumPy, SciPy, scikit-learn, UMAP,
statsmodels, and matplotlib, which are installed automatically with FlowMap.

You do not need a special single-cell environment to use the core API.

Optional Single-Cell Environment
--------------------------------

For RNA velocity workflows, it is often useful to work in a fresh environment
with common single-cell tools installed.

Using conda:

.. code-block:: bash

   conda create -n flowmap python=3.11
   conda activate flowmap

Using mamba:

.. code-block:: bash

   mamba create -n flowmap python=3.11
   mamba activate flowmap

Then install FlowMap and optional RNA velocity tools:

.. code-block:: bash

   pip install git+https://github.com/jingyuanhu01/flowmap.git
   pip install scanpy anndata scvelo

Scanpy, AnnData, and scVelo are not required by the core FlowMap API, but they
are commonly used for loading, preprocessing, and exploring single-cell RNA
velocity data.

Development Install
-------------------

If you want to edit the source code locally:

.. code-block:: bash

   git clone https://github.com/jingyuanhu01/flowmap.git
   cd flowmap
   pip install -e .
