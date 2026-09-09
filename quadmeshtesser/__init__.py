"""SWC tree skeleton → quadrilateral mesh (convolution surface pipeline)."""

from quadmeshtesser.junction_methods import (
    DEFAULT_BRANCH_JUNCTION,
    JunctionMethod,
    parse_branch_junction,
)
from quadmeshtesser.pipeline import PipelineParams, TreeQuadPipeline

__all__ = [
    "DEFAULT_BRANCH_JUNCTION",
    "JunctionMethod",
    "PipelineParams",
    "TreeQuadPipeline",
    "parse_branch_junction",
    "__version__",
]
__version__ = "0.2.0"
