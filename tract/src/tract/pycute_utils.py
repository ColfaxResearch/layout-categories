"""
Backward-compatible shim for the pycute layout bridge.

The implementation lives in :mod:`tract.backends.pycute`.
"""

from .backends.pycute import (
    _flat,
    flatten_layout,
    flat_modes,
    layouts_agree,
    compose_layouts,
    coalesce_layout,
    logical_product_layouts,
    nullify_trivial_strides,
    nullify_zero_strides,
    sort_flat_layout,
    sort_flat_layout_with_perm,
    is_tractable,
    compute_flat_layout_components,
    compute_flat_layout,
    compute_layout,
    flat_concatenate,
    concatenate,
    flat_complement,
    compute_Tuple_morphism,
    compute_Nest_morphism,
    compute_morphism,
)
