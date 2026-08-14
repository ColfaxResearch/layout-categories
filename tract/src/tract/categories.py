"""Backward-compatible aggregation of the core category modules.

The implementations live in :mod:`tract.fin`, :mod:`tract.nested_tuple`,
:mod:`tract.tuple_morphism`, and :mod:`tract.nest_morphism`.
"""

from .fin import FinMorphism
from .nested_tuple import NestedTuple
from .tuple_morphism import TupleMorphism
from .nest_morphism import (
    NestMorphism,
    make_morphism,
    compose,
    coalesce,
    complement,
    logical_divide,
    logical_product,
    morphism_to_tikz,
)

# Deprecated aliases (paper-era names)
Fin_morphism = FinMorphism
Tuple_morphism = TupleMorphism
Nest_morphism = NestMorphism
