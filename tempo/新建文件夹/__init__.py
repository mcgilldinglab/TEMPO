# tempo_sc/__init__.py

"""
TEMPO-SC: Time-Point Selection for Single-Cell & Bulk Temporal Data

This package contains:
- TEMPO_Selector: main class for beam-search time point selection
- TinyAE, Regressor, GeneGCN: neural network building blocks
- sinusoidal_pe, set_seed: core utility functions
"""

from .selector import TEMPO_Selector

from .models import (
    TinyAE,
    Regressor,
    GeneGCN,
)

from .utils import (
    sinusoidal_pe,
    set_seed,
)

__all__ = [
    "TEMPO_Selector",
    "TinyAE",
    "Regressor",
    "GeneGCN",
    "sinusoidal_pe",
    "set_seed",
]
