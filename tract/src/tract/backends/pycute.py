"""
pycute layout backend: bridges between tract morphisms and layouts from
NVIDIA's pycute (https://github.com/NVlabs/CuTe), the official pure-Python
reference implementation of CuTe. Requires no GPU and no cutlass DSL; the
test suite uses it to cross-validate tract's categorical operations against
the CuTe layout algebra.

pycute is not a PyPI dependency of tract; install it from a local clone,
e.g. ``uv pip install -e /path/to/CuTe``.
"""

from pycute import Layout, coalesce, composition, logical_product

from ..nested_tuple import NestedTuple
from ..nest_morphism import NestMorphism
from ..tuple_morphism import TupleMorphism
from . import base

_flat = base.flatten_nested


def flatten_layout(layout: Layout) -> Layout:
    """Compute the flattening of a layout."""
    return Layout(_flat(layout.shape), _flat(layout.stride))


def flat_modes(layout: Layout) -> tuple:
    """
    The flat tuple of (shape, stride) modes of a layout, with the stride of
    every size-1 mode nullified and the empty layout normalized to ((1, 0),).

    Two flat layouts define the same function on coordinates mode-by-mode
    exactly when their flat_modes agree, regardless of whether pycute
    represents them with int or tuple shapes.
    """
    shape = _flat(layout.shape)
    stride = _flat(layout.stride)
    modes = tuple((s, d if s != 1 else 0) for s, d in zip(shape, stride))
    return modes if modes else ((1, 0),)


def layouts_agree(layout1: Layout, layout2: Layout) -> bool:
    """Check whether two layouts agree mode-by-mode after flattening."""
    return flat_modes(layout1) == flat_modes(layout2)


def compose_layouts(A: Layout, B: Layout) -> Layout:
    """
    The layout composition A ∘ B via pycute, working around a pycute edge
    case: composition with a rank-0 right factor crashes in
    pycute.make_layout, whereas C++ CuTe returns the empty layout.
    """
    if B.shape == ():
        return Layout((), ())
    return composition(A, B)


def coalesce_layout(A: Layout, profile=1) -> Layout:
    """
    Coalesce a layout via pycute, working around a pycute edge case: an
    empty target profile crashes in pycute.make_layout, whereas C++ CuTe
    returns the empty layout (the only layout with the empty profile).
    """
    if profile == ():
        return Layout((), ())
    return coalesce(A, profile)


def logical_product_layouts(A: Layout, B: Layout) -> Layout:
    """
    The logical product of layouts via pycute, working around a pycute edge
    case: a rank-0 second factor crashes in pycute.make_layout, whereas C++
    CuTe returns (A, ():()) since the composition of complement(A) with the
    empty layout is empty.
    """
    if B.shape == ():
        return Layout((A.shape, ()), (A.stride, ()))
    return logical_product(A, B)


def nullify_trivial_strides(flat_layout: Layout) -> Layout:
    """Set the stride of every shape-1 mode of a flat layout to 0."""
    shape = flat_layout.shape
    stride = flat_layout.stride
    new_stride = tuple(
        stride[i] if shape[i] != 1 else 0 for i in range(len(shape))
    )
    return Layout(shape, new_stride)


def nullify_zero_strides(layout: Layout) -> Layout:
    """Set the stride of every shape-1 mode of a nested layout to 0."""
    flat_layout = nullify_trivial_strides(flatten_layout(layout))
    shape = NestedTuple(layout.shape).sub(flat_layout.shape).data
    stride = NestedTuple(layout.stride).sub(flat_layout.stride).data
    return Layout(shape, stride)


def sort_flat_layout(flat_layout: Layout) -> Layout:
    """Sort a flat layout by stride values, breaking ties by shape."""
    if len(flat_layout.shape) == 0:
        return flat_layout
    shape, stride = base.sorted_modes(flat_layout.shape, flat_layout.stride)
    return Layout(shape, stride)


def sort_flat_layout_with_perm(flat_layout: Layout):
    """Sort a flat layout and return the 1-indexed permutation used."""
    if len(flat_layout.shape) == 0:
        return flat_layout, []
    shape, stride, permutation = base.sorted_modes_with_perm(
        flat_layout.shape, flat_layout.stride
    )
    return Layout(shape, stride), permutation


def is_tractable(layout: Layout) -> bool:
    """Check whether a layout is tractable (see base.is_tractable_modes)."""
    return base.is_tractable_modes(_flat(layout.shape), _flat(layout.stride))


def compute_flat_layout_components(
    morphism: TupleMorphism,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The (shape, stride) pair of the layout L_f of a tuple morphism f.

    No 32-bit overflow check is performed: pycute works with
    arbitrary-precision Python integers.
    """
    return base.flat_layout_components(morphism, check_overflow=False)


def compute_flat_layout(morphism: TupleMorphism) -> Layout:
    """The layout L_f associated to a tuple morphism f."""
    shape_tuple, stride_tuple = compute_flat_layout_components(morphism)
    return Layout(shape_tuple, stride_tuple)


def compute_layout(morphism: NestMorphism) -> Layout:
    """The layout associated to a nested tuple morphism."""
    flat_layout = compute_flat_layout(morphism.flatten())
    shape = morphism.domain.data
    stride = morphism.domain.sub(flat_layout.stride).data
    return Layout(shape, stride)


def flat_concatenate(base_layout: Layout, stack: Layout) -> Layout:
    """The flat concatenation concat(L_1, L_2) of flat layouts."""
    return Layout(
        _flat(base_layout.shape) + _flat(stack.shape),
        _flat(base_layout.stride) + _flat(stack.stride),
    )


def concatenate(base_layout: Layout, stack: Layout) -> Layout:
    """The nested concatenation (L_1, L_2) of layouts."""
    return Layout(
        (base_layout.shape, stack.shape),
        (base_layout.stride, stack.stride),
    )


def flat_complement(flat_layout: Layout, N: int) -> Layout:
    """The complement of a flat layout with respect to total size N."""
    shape, stride = base.flat_complement_components(
        flat_layout.shape, flat_layout.stride, N
    )
    return Layout(shape, stride)


def compute_Tuple_morphism(flat_layout: Layout) -> TupleMorphism:
    """
    The standard representation f_L of a tractable flat layout L.

    :raises ValueError: If the layout is not tractable
    """
    return base.standard_tuple_morphism(
        tuple(flat_layout.shape), tuple(flat_layout.stride)
    )


def compute_Nest_morphism(layout: Layout) -> NestMorphism:
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
