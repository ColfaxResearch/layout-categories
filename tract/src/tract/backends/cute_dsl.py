"""
CuTe DSL layout backend: bridges between tract morphisms and
``cutlass.cute.Layout`` objects. Requires the ``nvidia-cutlass-dsl``
package.
"""

import cutlass
import cutlass.cute as cute

from ..nested_tuple import NestedTuple
from ..nest_morphism import NestMorphism
from ..tuple_morphism import TupleMorphism
from . import base


def nullify_trivial_strides(flat_layout: cute.Layout) -> cute.Layout:
    """Set the stride of every shape-1 mode of a flat layout to 0."""
    shape = flat_layout.shape
    stride = flat_layout.stride
    new_stride = tuple(
        stride[i] if shape[i] != 1 else 0 for i in range(len(shape))
    )
    return cute.make_layout(shape, stride=new_stride)


def nullify_zero_strides(layout: cute.Layout) -> cute.Layout:
    """Set the stride of every shape-1 mode of a nested layout to 0."""
    flat_layout = nullify_trivial_strides(flatten_layout(layout))
    shape = NestedTuple(layout.shape).sub(flat_layout.shape).data
    stride = NestedTuple(layout.stride).sub(flat_layout.stride).data
    return cute.make_layout(shape, stride=stride)


def flatten_layout(layout: cute.Layout) -> cute.Layout:
    """Compute the flattening of a layout."""
    return cute.make_layout(
        cute.flatten_to_tuple(layout.shape),
        stride=cute.flatten_to_tuple(layout.stride),
    )


def sort_flat_layout(flat_layout: cute.Layout) -> cute.Layout:
    """Sort a flat layout by stride values, breaking ties by shape."""
    if len(flat_layout.shape) == 0:
        return flat_layout
    shape, stride = base.sorted_modes(flat_layout.shape, flat_layout.stride)
    return cute.make_layout(shape, stride=stride)


def sort_flat_layout_with_perm(flat_layout: cute.Layout):
    """Sort a flat layout and return the 1-indexed permutation used."""
    if len(flat_layout.shape) == 0:
        return flat_layout, []
    shape, stride, permutation = base.sorted_modes_with_perm(
        flat_layout.shape, flat_layout.stride
    )
    return cute.make_layout(shape, stride=stride), permutation


def is_tractable(layout: cute.Layout) -> bool:
    """Check whether a layout is tractable (see base.is_tractable_modes)."""
    flat_layout = flatten_layout(layout)
    return base.is_tractable_modes(
        base.flatten_nested(flat_layout.shape),
        base.flatten_nested(flat_layout.stride),
    )


@cute.jit
def compute_Tuple_morphism(flat_layout: cute.Layout) -> TupleMorphism:
    """
    The standard representation f_L of a tractable flat layout L.

    :raises ValueError: If the layout is not tractable
    """
    if cutlass.const_expr(is_tractable(flat_layout)):
        return base.standard_tuple_morphism(
            tuple(flat_layout.shape), tuple(flat_layout.stride)
        )
    else:
        raise ValueError("The provided layout is not tractable.")


def compute_flat_layout_components(
    morphism: TupleMorphism,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The (shape, stride) pair of the layout L_f of a tuple morphism f."""
    return base.flat_layout_components(morphism, check_overflow=True)


def compute_flat_layout(morphism: TupleMorphism) -> cute.Layout:
    """The layout L_f associated to a tuple morphism f."""
    shape_tuple, stride_tuple = compute_flat_layout_components(morphism)
    return cute.make_layout(shape_tuple, stride=stride_tuple)


def flat_concatenate(base_layout: cute.Layout, stack: cute.Layout) -> cute.Layout:
    """The flat concatenation concat(L_1, L_2) of flat layouts."""
    def intuple_to_tuple(shape):
        if isinstance(shape, int):
            return (shape,)
        return shape

    concat_shape = intuple_to_tuple(base_layout.shape) + intuple_to_tuple(stack.shape)
    concat_stride = intuple_to_tuple(base_layout.stride) + intuple_to_tuple(
        stack.stride
    )
    return cute.make_layout(concat_shape, stride=concat_stride)


def concatenate(base_layout: cute.Layout, stack: cute.Layout) -> cute.Layout:
    """The nested concatenation (L_1, L_2) of layouts."""
    return cute.make_layout(
        (base_layout.shape, stack.shape),
        stride=(base_layout.stride, stack.stride),
    )


def compute_layout(morphism: NestMorphism) -> cute.Layout:
    """The layout associated to a nested tuple morphism."""
    flat_layout = compute_flat_layout(morphism.flatten())
    shape = morphism.domain.data
    stride = morphism.domain.sub(flat_layout.stride).data
    return cute.make_layout(shape, stride=stride)


@cute.jit
def compute_Nest_morphism(layout: cute.Layout) -> NestMorphism:
    """
    A nested tuple morphism f with L_f = L, for a tractable layout L.

    :raises ValueError: If the layout is not tractable
    """
    flat_layout = flatten_layout(layout)
    flat_morphism = compute_Tuple_morphism(flat_layout)

    domain = NestedTuple(layout.shape)
    codomain = NestedTuple(flat_morphism.codomain)
    return NestMorphism(domain, codomain, flat_morphism.map)


compute_morphism = compute_Nest_morphism


def flat_complement(flat_layout: cute.Layout, N: int) -> cute.Layout:
    """The complement of a flat layout with respect to total size N."""
    shape, stride = base.flat_complement_components(
        flat_layout.shape, flat_layout.stride, N
    )
    return cute.make_layout(shape, stride=stride)


def layout_to_tikz(layout: cute.Layout, full_doc=False) -> str:
    """Render a tractable layout's nested morphism as a TikZ diagram."""
    from ..tuple_morph_tikz import nested_tuple_morphism_to_tikz

    morphism = compute_Nest_morphism(layout)
    layout_str = str(layout)
    return nested_tuple_morphism_to_tikz(
        morphism,
        row_spacing=0.8,
        tree_width=2.2,
        map_width=3.0,
        root_y_offset=0.0,
        label=f"${layout_str}$",
        full_doc=full_doc,
    )
