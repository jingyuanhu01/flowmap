"""
Geometry submodule.

Provides differential geometric analysis tools for
vector-field embeddings.
"""

from .fixed_points import FixedPointAnalyzer
from .gradient_analyzer import GeneGradientAnalyzer
from .least_action_path import LagrangianPathOptimizer
from .curvature import compute_flow_curvature

__all__ = [
    "FixedPointAnalyzer",
    "GeneGradientAnalyzer",
    "LagrangianPathOptimizer",
    "compute_flow_curvature",
]
