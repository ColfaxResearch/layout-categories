"""
Backward-compatible shim for the CuTe DSL layout bridge.

The implementations live in :mod:`tract.backends.cute_dsl` (layout ↔
morphism bridge, requires the cutlass DSL) and :mod:`tract.refinement`
(mutual refinement and weak composition, pure).
"""

from .backends.cute_dsl import (
    nullify_trivial_strides,
    nullify_zero_strides,
    flatten_layout,
    sort_flat_layout,
    sort_flat_layout_with_perm,
    is_tractable,
    compute_Tuple_morphism,
    compute_flat_layout_components,
    compute_flat_layout,
    flat_concatenate,
    concatenate,
    compute_layout,
    compute_Nest_morphism,
    compute_morphism,
    flat_complement,
    layout_to_tikz,
)
from .refinement import (
    mutual_refinement,
    weak_composite,
    mutual_refinement_to_tikz,
)
