from .flowmap_embedding import VectorFieldEmbedder
from . import core
from . import plot
from . import geometry
from . import evaluation
from . import utils

__all__ = [
    "VectorFieldEmbedder",
    "core",
    "plot",
    "geometry",
    "evaluation",
    "utils",
]

__version__ = "0.3.0"

