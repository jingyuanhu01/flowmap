import os
import sys
sys.path.insert(0, os.path.abspath("../src"))

project = "FlowMap"
author = "Jingyuan Hu"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "sphinx_rtd_theme"
