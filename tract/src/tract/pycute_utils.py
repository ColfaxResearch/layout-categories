"""
pycute-backed layout utilities for the Tract library.

Pure-Python analogues of the functions in layout_utils.py, built on NVIDIA's
pycute (https://github.com/NVlabs/CuTe), the official Python reference
implementation of CuTe. Unlike layout_utils.py, this module requires no GPU
and no cutlass DSL; it is used by the test suite to cross-validate tract's
categorical operations against the CuTe layout algebra.

pycute is not a PyPI dependency of tract; install it from a local clone,
e.g. ``uv pip install -e /path/to/CuTe``.
"""

from pycute import Layout, coalesce, composition, logical_product

from .categories import (
    Tuple_morphism,
    Nest_morphism,
    NestedTuple,
)


def _flat(x) -> tuple:
    """
    Flatten an int or arbitrarily nested tuple to a flat tuple of leaves.

    :param x: Integer or nested tuple
    :return: Flat tuple of leaves
    :rtype: tuple
    """
    if isinstance(x, tuple):
        return tuple(leaf for entry in x for leaf in _flat(entry))
    return (x,)


def flatten_layout(layout: Layout) -> Layout:
    """
    Compute the flattening of a given layout.

    :param layout: Input layout
    :type layout: Layout
    :return: Flattened layout
    :rtype: Layout
    """
    return Layout(_flat(layout.shape), _flat(layout.stride))


def flat_modes(layout: Layout) -> tuple:
    """
    The flat tuple of (shape, stride) modes of a layout, with the stride of
    every size-1 mode nullified and the empty layout normalized to ((1, 0),).

    Two flat layouts define the same function on coordinates mode-by-mode
    exactly when their flat_modes agree, regardless of whether pycute
    represents them with int or tuple shapes.

    :param layout: Input layout
    :type layout: Layout
    :return: Tuple of (shape, stride) pairs
    :rtype: tuple
    """
    shape = _flat(layout.shape)
    stride = _flat(layout.stride)
    modes = tuple((s, d if s != 1 else 0) for s, d in zip(shape, stride))
    return modes if modes else ((1, 0),)


def layouts_agree(layout1: Layout, layout2: Layout) -> bool:
    """
    Check whether two layouts agree mode-by-mode after flattening, ignoring
    strides of size-1 modes and int-vs-tuple shape representation.

    :param layout1: First layout
    :type layout1: Layout
    :param layout2: Second layout
    :type layout2: Layout
    :return: True if the flattened modes agree
    :rtype: bool
    """
    return flat_modes(layout1) == flat_modes(layout2)


def compose_layouts(A: Layout, B: Layout) -> Layout:
    """
    Compute the layout composition A ∘ B via pycute, working around a pycute
    edge case: composition with a rank-0 right factor crashes in
    pycute.make_layout, whereas C++ CuTe returns the empty layout.

    :param A: Outer layout
    :type A: Layout
    :param B: Inner layout (its domain is the composite's domain)
    :type B: Layout
    :return: The composition A ∘ B
    :rtype: Layout
    """
    if B.shape == ():
        return Layout((), ())
    return composition(A, B)


def coalesce_layout(A: Layout, profile=1) -> Layout:
    """
    Coalesce a layout via pycute, working around a pycute edge case: an
    empty target profile crashes in pycute.make_layout, whereas C++ CuTe
    returns the empty layout (the only layout with the empty profile).

    :param A: Layout to coalesce
    :type A: Layout
    :param profile: Target profile, as in pycute.coalesce
    :return: The coalesced layout
    :rtype: Layout
    """
    if profile == ():
        return Layout((), ())
    return coalesce(A, profile)


def logical_product_layouts(A: Layout, B: Layout) -> Layout:
    """
    Compute the logical product of layouts via pycute, working around a
    pycute edge case: a rank-0 second factor crashes in pycute.make_layout,
    whereas C++ CuTe returns (A, ():()) since the composition of
    complement(A) with the empty layout is empty.

    :param A: First factor
    :type A: Layout
    :param B: Second factor
    :type B: Layout
    :return: The logical product of A and B
    :rtype: Layout
    """
    if B.shape == ():
        return Layout((A.shape, ()), (A.stride, ()))
    return logical_product(A, B)


def nullify_trivial_strides(flat_layout: Layout) -> Layout:
    """
    Set stride to 0 for any dimension with shape 1.

    :param flat_layout: Input flat layout
    :type flat_layout: Layout
    :return: Layout with nullified trivial strides
    :rtype: Layout
    """
    shape = flat_layout.shape
    stride = flat_layout.stride
    new_stride = tuple(
        stride[i] if shape[i] != 1 else 0 for i in range(len(shape))
    )
    return Layout(shape, new_stride)


def nullify_zero_strides(layout: Layout) -> Layout:
    """
    Nullify strides for dimensions with shape 1 in nested layouts.

    :param layout: Input layout
    :type layout: Layout
    :return: Layout with nullified zero strides
    :rtype: Layout
    """
    flat_layout = nullify_trivial_strides(flatten_layout(layout))
    shape = NestedTuple(layout.shape).sub(flat_layout.shape).data
    stride = NestedTuple(layout.stride).sub(flat_layout.stride).data
    return Layout(shape, stride)


def sort_flat_layout(flat_layout: Layout) -> Layout:
    """
    Sort a flat layout by stride values, breaking ties by shape.

    :param flat_layout: Input flat layout
    :type flat_layout: Layout
    :return: Sorted layout
    :rtype: Layout
    """
    if len(flat_layout.shape) == 0:
        return flat_layout

    indexed = list(zip(flat_layout.shape, flat_layout.stride))
    sorted_pairs = sorted(indexed, key=lambda x: (x[1], x[0]))
    sorted_shape, sorted_stride = zip(*sorted_pairs)
    return Layout(tuple(sorted_shape), tuple(sorted_stride))


def sort_flat_layout_with_perm(flat_layout: Layout):
    """
    Sort a flat layout and return the permutation used.

    :param flat_layout: Input flat layout
    :type flat_layout: Layout
    :return: Tuple of (sorted layout, permutation)
    :rtype: Tuple[Layout, list]
    """
    if len(flat_layout.shape) == 0:
        return flat_layout, []

    indexed = list(enumerate(zip(flat_layout.shape, flat_layout.stride)))
    sorted_indexed = sorted(indexed, key=lambda x: (x[1][1], x[1][0]))
    permutation = [index + 1 for index, _ in sorted_indexed]
    sorted_shape, sorted_stride = zip(*[item for _, item in sorted_indexed])
    sorted_layout = Layout(tuple(sorted_shape), tuple(sorted_stride))
    return sorted_layout, permutation


def is_tractable(layout: Layout) -> bool:
    """
    Check if a given layout is tractable.

    A layout is tractable if each stride divides evenly into the next
    stride times shape product.

    :param layout: Input layout
    :type layout: Layout
    :return: True if tractable
    :rtype: bool
    """
    flat_layout = flatten_layout(layout)
    sorted_flat_layout = sort_flat_layout(flat_layout)
    shape = sorted_flat_layout.shape
    stride = sorted_flat_layout.stride

    for i in range(len(shape) - 1):
        if stride[i] != 0:
            if stride[i + 1] % (shape[i] * stride[i]) != 0:
                return False
    return True


def compute_flat_layout_components(
    morphism: Tuple_morphism,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the ``(shape, stride)`` pair associated to a tuple morphism.

    Identical to layout_utils.compute_flat_layout_components, except that no
    32-bit overflow check is performed: pycute works with arbitrary-precision
    Python integers.
    """
    domain = morphism.domain
    codomain = morphism.codomain
    alpha = morphism.map

    m = len(domain)
    stride_list = [0] * m

    for i in range(m):
        if alpha[i] != 0:
            t = 1
            for j in range(alpha[i] - 1):
                t *= codomain[j]
            stride_list[i] = t

    return tuple(domain), tuple(stride_list)


def compute_flat_layout(morphism: Tuple_morphism) -> Layout:
    """
    Compute the layout L_f associated to a tuple morphism f.

    :param morphism: Input tuple morphism
    :type morphism: Tuple_morphism
    :return: Corresponding layout
    :rtype: Layout
    """
    shape_tuple, stride_tuple = compute_flat_layout_components(morphism)
    return Layout(shape_tuple, stride_tuple)


def compute_layout(morphism: Nest_morphism) -> Layout:
    """
    Compute the layout associated to a nested tuple morphism.

    :param morphism: Input nested tuple morphism
    :type morphism: Nest_morphism
    :return: Corresponding layout
    :rtype: Layout
    """
    flat_layout = compute_flat_layout(morphism.flatten())
    shape = morphism.domain.data
    stride = morphism.domain.sub(flat_layout.stride).data
    return Layout(shape, stride)


def flat_concatenate(base: Layout, stack: Layout) -> Layout:
    """
    Compute the flat concatenation concat(L_1, L_2) of flat layouts.

    :param base: First layout
    :type base: Layout
    :param stack: Second layout
    :type stack: Layout
    :return: Concatenated layout
    :rtype: Layout
    """
    return Layout(
        _flat(base.shape) + _flat(stack.shape),
        _flat(base.stride) + _flat(stack.stride),
    )


def concatenate(base: Layout, stack: Layout) -> Layout:
    """
    Compute the nested concatenation (L_1, L_2) of layouts.

    :param base: First layout
    :type base: Layout
    :param stack: Second layout
    :type stack: Layout
    :return: Nested concatenation
    :rtype: Layout
    """
    return Layout(
        (base.shape, stack.shape),
        (base.stride, stack.stride),
    )


def flat_complement(flat_layout: Layout, N: int) -> Layout:
    """
    Compute the complement of a flat layout with respect to size N.

    :param flat_layout: Input flat layout
    :type flat_layout: Layout
    :param N: Total size
    :type N: int
    :return: Complement layout
    :rtype: Layout
    """
    reduced_layout = sort_flat_layout(flat_layout)
    S = reduced_layout.shape
    D = reduced_layout.stride
    m = len(S)

    shape = [D[0]]
    for i in range(1, m):
        shape.append(D[i] // (S[i - 1] * D[i - 1]))
    shape.append(N // (S[-1] * D[-1]))

    stride = [1]
    for i in range(m):
        stride.append(S[i] * D[i])

    return Layout(tuple(shape), tuple(stride))


def compute_Tuple_morphism(flat_layout: Layout) -> Tuple_morphism:
    """
    Compute a tuple morphism from a tractable flat layout.

    Given a tractable flat layout L, produces the standard representation
    f_L of L. Pure-Python port of layout_utils.compute_Tuple_morphism.

    :param flat_layout: Input tractable flat layout
    :type flat_layout: Layout
    :return: Corresponding tuple morphism
    :rtype: Tuple_morphism
    :raises ValueError: If the layout is not tractable
    """
    if not is_tractable(flat_layout):
        raise ValueError("The provided layout is not tractable.")

    domain = tuple(flat_layout.shape)
    sorted_flat_layout, permutation = sort_flat_layout_with_perm(flat_layout)
    shape = tuple(sorted_flat_layout.shape)
    stride = tuple(sorted_flat_layout.stride)
    m = len(shape)

    # Find the largest integer k such that stride[k-1] = 0
    k = 0
    for entry in stride:
        if entry == 0:
            k += 1
        else:
            break

    # Build codomain
    codomain = tuple()
    if k < m:
        cod = [stride[k], shape[k]]
        for j in range(k + 1, m):
            denom = shape[j - 1] * stride[j - 1]
            factor = (stride[j] // denom) if denom != 0 else 0
            cod.append(int(factor))
            cod.append(shape[j])
        codomain = tuple(cod)

    # Construct the map alpha'
    alpha_prime = [0] * m
    for j in range(k, m):
        alpha_prime[j] = 2 * (j - k + 1)

    # Construct the inverse permutation
    inverse_permutation = [0] * m
    for i in range(m):
        inverse_permutation[permutation[i] - 1] = i + 1

    # alpha = alpha'[σ^{-1}(i)]
    alpha = tuple(alpha_prime[inverse_permutation[i] - 1] for i in range(m))

    morphism = Tuple_morphism(domain, codomain, alpha)
    restricted_codomain_indices = []
    for i in range(len(codomain)):
        if codomain[i] != 1 or i + 1 in morphism.map:
            restricted_codomain_indices.append(i + 1)
    return morphism.factorize(tuple(restricted_codomain_indices))


def compute_Nest_morphism(layout: Layout) -> Nest_morphism:
    """
    Compute a nested tuple morphism from a tractable layout.

    Given a tractable layout L, produces a nested tuple morphism f with
    L_f = L. Pure-Python port of layout_utils.compute_Nest_morphism.

    :param layout: Input tractable layout
    :type layout: Layout
    :return: Corresponding nested tuple morphism
    :rtype: Nest_morphism
    """
    flat_layout = flatten_layout(layout)
    flat_morphism = compute_Tuple_morphism(flat_layout)

    domain = NestedTuple(layout.shape)
    codomain = NestedTuple(flat_morphism.codomain)
    return Nest_morphism(domain, codomain, flat_morphism.map)


compute_morphism = compute_Nest_morphism
