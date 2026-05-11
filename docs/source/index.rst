FlowMap
=======

.. raw:: html

   <section class="flowmap-hero">
     <p class="flowmap-kicker">Geometry-aware vector field embeddings</p>
     <p class="flowmap-lede">
       FlowMap embeds high-dimensional expression and velocity data into a
       smooth low-dimensional geometry where cell-state flow can be visualized,
       compared, and analyzed.
     </p>
     <div class="flowmap-actions">
       <a href="installation.html">Install</a>
       <a href="tutorials/index.html">Tutorials</a>
     </div>
   </section>

.. figure:: _static/figures/fig2.png
   :alt: FlowMap overview schematic
   :class: flowmap-main-figure

   FlowMap integrates expression and velocity, fits smooth manifold geometry,
   maps vector fields into embedding space, and supports downstream dynamical
   analysis.

.. raw:: html

   <section class="flowmap-capabilities">
     <div>
       <h2>Embed flow</h2>
       <p>Construct embeddings that respect both cell-state similarity and velocity direction.</p>
     </div>
     <div>
       <h2>Map velocity</h2>
       <p>Use splines and Jacobians to project high-dimensional dynamics into embedding space.</p>
     </div>
     <div>
       <h2>Analyze geometry</h2>
       <p>Study trajectories, fixed points, curvature, and gene-level gradients.</p>
     </div>
   </section>

.. toctree::
   :maxdepth: 2
   :hidden:

   introduction
   installation
   tutorials/index
   api
   references
