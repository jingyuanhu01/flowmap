from .flowmap_embedding import VectorFieldEmbedder
from .core.spline import Spline

# Expose submodules
from . import core
from . import plot
from . import geometry
from . import evaluation
from . import utils

__all__ = [
    "VectorFieldEmbedder",
    "Spline",
    "core",
    "plot",
    "geometry",
    "evaluation",
    "utils",
]

__version__ = "0.3.0"

