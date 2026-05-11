import os
import sys
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("../_build/matplotlib"))
sys.path.insert(0, os.path.abspath("../../src"))

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
autodoc_mock_imports = ["umap"]

html_theme = "furo"
html_title = "FlowMap"
html_static_path = ["_static"]
html_css_files = ["flowmap.css"]
html_theme_options = {
    "light_logo": "flowmap_logo.png",
    "dark_logo": "flowmap_logo_dark.png",
}
