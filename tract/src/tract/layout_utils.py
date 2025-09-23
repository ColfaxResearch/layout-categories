"""
Layout computation utilities for Tract library.

This module provides functions to convert between category-theoretic morphisms
and CuTe layouts, enabling GPU kernel optimizations.
"""

import cutlass
import cutlass.cute as cute
from tract.tuple_morph_tikz import nested_tuple_morphism_to_tikz, two_parenthesizations_to_tikz_values

from .categories import (
    Fin_morphism,
    Tuple_morphism,
    Nest_morphism,
    NestedTuple
)


def nullify_trivial_strides(flat_layout: cute.Layout) -> cute.Layout:
    """
    Set stride to 0 for any dimension with shape 1.
    
    Given a flat layout L = (s_1,...,s_m):(d_1,...,d_m),
    sets d_i = 0 if s_i = 1.
    
    :param flat_layout: Input layout
    :type flat_layout: cute.Layout
    :return: Layout with nullified trivial strides
    :rtype: cute.Layout
    """
    shape = flat_layout.shape
    stride = flat_layout.stride
    new_stride = []
    
    for i in range(len(shape)):
        if shape[i] != 1:
            new_stride.append(stride[i])
        else:
            new_stride.append(0)
            
    return cute.make_layout(shape, stride=tuple(new_stride))


def nullify_zero_strides(layout: cute.Layout) -> cute.Layout:
    """
    Nullify strides for dimensions with shape 1 in nested layouts.
    
    :param layout: Input layout
    :type layout: cute.Layout
    :return: Layout with nullified zero strides
    :rtype: cute.Layout
    """
    flat_layout = nullify_trivial_strides(flatten_layout(layout))
    shape = NestedTuple(layout.shape).sub(flat_layout.shape).data
    stride = NestedTuple(layout.stride).sub(flat_layout.stride).data
    return cute.make_layout(shape, stride=stride)


def flatten_layout(layout: cute.Layout) -> cute.Layout:
    """
    Compute the flattening of a given layout.
    
    :param layout: Input layout
    :type layout: cute.Layout
    :return: Flattened layout
    :rtype: cute.Layout
    """
    flat_layout = cute.make_layout(
        cute.flatten_to_tuple(layout.shape), 
        stride=cute.flatten_to_tuple(layout.stride)
    )
    return flat_layout


def sort_flat_layout(flat_layout: cute.Layout) -> cute.Layout:
    """
    Sort a flat layout by stride values, breaking ties by shape.
    
    :param flat_layout: Input flat layout
    :type flat_layout: cute.Layout
    :return: Sorted layout
    :rtype: cute.Layout
    """
    if len(flat_layout.shape) == 0:
        return flat_layout
        
    indexed = list(zip(flat_layout.shape, flat_layout.stride))
    sorted_pairs = sorted(indexed, key=lambda x: (x[1], x[0]))
    sorted_shape, sorted_stride = zip(*sorted_pairs)
    return cute.make_layout(sorted_shape, stride=sorted_stride)


def sort_flat_layout_with_perm(flat_layout: cute.Layout):
    """
    Sort a flat layout and return the permutation used.
    
    :param flat_layout: Input flat layout
    :type flat_layout: cute.Layout
    :return: Tuple of (sorted layout, permutation)
    :rtype: Tuple[cute.Layout, list]
    """
    if len(flat_layout.shape) == 0:
        return flat_layout, []
        
    indexed = list(enumerate(zip(flat_layout.shape, flat_layout.stride)))
    sorted_indexed = sorted(indexed, key=lambda x: (x[1][1], x[1][0]))
    permutation = [index + 1 for index, _ in sorted_indexed]
    sorted_shape, sorted_stride = zip(*[item for _, item in sorted_indexed])
    sorted_layout = cute.make_layout(sorted_shape, stride=sorted_stride)
    return sorted_layout, permutation


def is_tractable(layout: cute.Layout) -> bool:
    """
    Check if a given layout is tractable.
    
    A layout is tractable if each stride divides evenly into the next
    stride times shape product.
    
    :param layout: Input layout
    :type layout: cute.Layout
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


@cute.jit
def compute_Tuple_morphism(flat_layout: cute.Layout) -> Tuple_morphism:
    """
    Compute a tuple morphism from a tractable flat layout.
    
    Given a tractable flat layout L, produces the standard representation f_L of L.
    
    :param flat_layout: Input tractable flat layout
    :type flat_layout: cute.Layout
    :return: Corresponding tuple morphism
    :rtype: Tuple_morphism
    """

    if cutlass.const_expr(is_tractable(flat_layout)):
        domain = tuple(flat_layout.shape)
        sorted_flat_layout, permutation = sort_flat_layout_with_perm(flat_layout)
        shape = tuple(sorted_flat_layout.shape)
        stride = tuple(sorted_flat_layout.stride)
        m = len(shape)

        # Find the largest integer k such that stride[k-1] = 0
        k = 0
        seen_nonzero = False
        for i in cutlass.range_constexpr(len(stride)):
            if cutlass.const_expr(stride[i] == 0 and not seen_nonzero):
                k += 1
            else:
                seen_nonzero = True

        # Build codomain
        codomain = tuple()
        if cutlass.const_expr(k < m):
            cod = [stride[k], shape[k]]
            for j in cutlass.range_constexpr(k + 1, m):
                denom = shape[j - 1] * stride[j - 1]
                # Use // for exact integer division
                factor = (stride[j] // denom) if denom != 0 else 0
                cod.append(int(factor))
                cod.append(shape[j])
            codomain = tuple(cod)

        # Construct the map alpha'
        alpha_prime = [0] * m
        for j in cutlass.range_constexpr(k, m):
            alpha_prime[j] = 2 * (j - k + 1)

        # Construct the inverse permutation
        inverse_permutation = [0] * m
        for i in cutlass.range_constexpr(m):
            inverse_permutation[permutation[i] - 1] = i + 1

        # alpha = alpha'[σ^{-1}(i)]
        alpha = tuple(alpha_prime[inverse_permutation[i] - 1] for i in range(m))
        
        morphism = Tuple_morphism(domain, codomain, alpha)
        restricted_codomain_indices=[]
        for i in cutlass.range_constexpr(len(codomain)):
            if cutlass.const_expr(codomain[i] != 1 or i+1 in morphism.map):
                restricted_codomain_indices.append(i+1)
        restricted_codomain_indices = tuple(restricted_codomain_indices)
        result = morphism.factorize(restricted_codomain_indices)

        return result
    else:
        raise ValueError("The provided layout is not tractable.")


def compute_flat_layout(morphism: Tuple_morphism) -> cute.Layout:
    """
    Compute the layout L_f associated to a tuple morphism f.
    
    :param morphism: Input tuple morphism
    :type morphism: Tuple_morphism
    :return: Corresponding layout
    :rtype: cute.Layout
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
            if t > 2**31 - 1:  # Check for potential overflow
                raise OverflowError("Stride value exceeds 32-bit integer limit.")
            stride_list[i] = t

    shape_tuple = tuple(domain)
    stride_tuple = tuple(stride_list)
    return cute.make_layout(shape_tuple, stride=stride_tuple)


def flat_concatenate(base: cute.Layout, stack: cute.Layout) -> cute.Layout:
    """
    Compute the flat concatenation concat(L_1, L_2) of flat layouts.
    
    :param base: First layout
    :type base: cute.Layout
    :param stack: Second layout
    :type stack: cute.Layout
    :return: Concatenated layout
    :rtype: cute.Layout
    """
    # Convert shapes to tuples for proper concatenation
    def intuple_to_tuple(shape):
        if isinstance(shape, int):
            return (shape,)
        else:
            return shape

    base_shape_tuple = intuple_to_tuple(base.shape)
    stack_shape_tuple = intuple_to_tuple(stack.shape)
    base_stride_tuple = intuple_to_tuple(base.stride)
    stack_stride_tuple = intuple_to_tuple(stack.stride)

    concat_shape = base_shape_tuple + stack_shape_tuple
    concat_stride = base_stride_tuple + stack_stride_tuple
    return cute.make_layout(concat_shape, stride=concat_stride)


def concatenate(base: cute.Layout, stack: cute.Layout) -> cute.Layout:
    """
    Compute the nested concatenation (L_1, L_2) of layouts.
    
    :param base: First layout
    :type base: cute.Layout
    :param stack: Second layout
    :type stack: cute.Layout
    :return: Nested concatenation
    :rtype: cute.Layout
    """
    return cute.make_layout(
        (base.shape, stack.shape), 
        stride=(base.stride, stack.stride)
    )


def compute_layout(morphism: Nest_morphism) -> cute.Layout:
    """
    Compute the layout associated to a nested tuple morphism.
    
    :param morphism: Input nested tuple morphism
    :type morphism: Nest_morphism
    :return: Corresponding layout
    :rtype: cute.Layout
    """
    flat_layout = compute_flat_layout(morphism.flatten())
    shape = morphism.domain.data
    stride = morphism.domain.sub(flat_layout.stride).data
    return cute.make_layout(shape, stride=stride)


@cute.jit
def compute_Nest_morphism(layout: cute.Layout) -> Nest_morphism:
    """
    Compute a nested tuple morphism from a tractable layout.
    
    Given a tractable layout L, produces a nested tuple morphism f with L_f = L.
    
    :param layout: Input tractable layout
    :type layout: cute.Layout
    :return: Corresponding nested tuple morphism
    :rtype: Nest_morphism
    """
    flat_layout = flatten_layout(layout)
    flat_morphism = compute_Tuple_morphism(flat_layout)

    domain = NestedTuple(layout.shape)
    codomain = NestedTuple(flat_morphism.codomain)
    map = flat_morphism.map
    morphism = Nest_morphism(domain, codomain, map)
    return morphism

compute_morphism = compute_Nest_morphism

def layout_to_tikz(layout: cute.Layout, full_doc=False) -> str:
    morphism = compute_Nest_morphism(layout)
    layout_str = str(layout)
    tikz_str = nested_tuple_morphism_to_tikz(
        morphism,
        row_spacing=0.8,
        tree_width=2.2,
        map_width=3.0,
        root_y_offset=0.0,
        label=f"${layout_str}$",
        full_doc=full_doc,
    )
    return tikz_str

def flat_complement(flat_layout: cute.Layout, N: int) -> cute.Layout:
    """
    Compute the complement of a flat layout with respect to size N.
    
    :param flat_layout: Input flat layout
    :type flat_layout: cute.Layout
    :param N: Total size
    :type N: int
    :return: Complement layout
    :rtype: cute.Layout
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
        
    return cute.make_layout(tuple(shape), stride=tuple(stride))


def mutual_refinement(nestedtuple1: NestedTuple, nestedtuple2: NestedTuple):
    """
    Compute mutual refinement of two nested tuples.
    
    Given nested tuples T and U, computes T' and U' such that:
    1. T' refines T
    2. U' refines U
    3. T' divides U'
    
    Example:
        T = (6,6), U = (2,6,6) -> T' = ((2,3),(2,3)), U' = (2,(3,2),(3,2))
    
    :param nestedtuple1: First nested tuple
    :type nestedtuple1: NestedTuple
    :param nestedtuple2: Second nested tuple
    :type nestedtuple2: NestedTuple
    :return: Tuple of refined nested tuples (T', U')
    :rtype: Tuple[NestedTuple, NestedTuple]
    :raises ValueError: If tuples are not mutually refinable
    """
    tuple1 = nestedtuple1.flatten()
    tuple2 = nestedtuple2.flatten()
    list1 = list(tuple1)
    list2 = list(tuple2)
    
    i = 0
    j = 0
    result1 = []
    cur_mode1 = []
    result2 = []
    cur_mode2 = []
    
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            cur_mode1.append(list1[i])
            result1.append(cur_mode1[0] if len(cur_mode1) == 1 else tuple(cur_mode1))
            cur_mode1 = []
            cur_mode2.append(list2[j])
            result2.append(cur_mode2[0] if len(cur_mode2) == 1 else tuple(cur_mode2))
            cur_mode2 = []
            i += 1
            j += 1
        elif list1[i] < list2[j] and list2[j] % list1[i] == 0:
            cur_mode1.append(list1[i])
            result1.append(cur_mode1[0] if len(cur_mode1) == 1 else tuple(cur_mode1))
            cur_mode1 = []
            cur_mode2.append(list1[i])
            list2[j] //= list1[i]
            i += 1
        elif list2[j] < list1[i] and list1[i] % list2[j] == 0:
            cur_mode1.append(list2[j])
            cur_mode2.append(list2[j])
            result2.append(cur_mode2[0] if len(cur_mode2) == 1 else tuple(cur_mode2))
            cur_mode2 = []
            list1[i] //= list2[j]
            j += 1
        else:
            raise ValueError("The given nested tuples are not mutually refinable.")
            
    if i < len(list1):
        raise ValueError("The given nested tuples are not mutually refinable.")
        
    if cur_mode2 != []:
        cur_mode2.append(list2[j])
        result2.append(tuple(cur_mode2))
        j += 1

    while j < len(list2):
        result2.append(list2[j])
        j += 1

    result1 = nestedtuple1.sub(tuple(result1))
    result2 = nestedtuple2.sub(tuple(result2))
    return result1, result2

def mutual_refinement_to_tikz(nestedtuple1: NestedTuple, nestedtuple2: NestedTuple) -> str:
    """
    Generate a TikZ diagram illustrating the mutual refinement of two nested tuples.
    
    :param nestedtuple1: First nested tuple
    :type nestedtuple1: NestedTuple
    :param nestedtuple2: Second nested tuple
    :type nestedtuple2: NestedTuple
    :return: TikZ diagram as a string
    :rtype: str
    """
    Tprime, Uprime = mutual_refinement(nestedtuple1, nestedtuple2)
    tikz_str = two_parenthesizations_to_tikz_values(
        Uprime.flatten(),
        Tprime.data,
        Uprime.data,
        row_spacing=0.8,
        left_width=2.5,
        right_width=2.5,
        root_y_offset=0.0,
        center_label=f"${Tprime} \\quad {Uprime}$",
        full_doc=False,
    )
    return tikz_str

def weak_composite(f: Nest_morphism, g: Nest_morphism) -> Nest_morphism:
    """
    Compute weak composition of nested tuple morphisms.
    
    Computes composition through mutual refinement when domains/codomains
    don't match exactly.
    
    :param f: First morphism
    :type f: Nest_morphism
    :param g: Second morphism
    :type g: Nest_morphism
    :return: Weak composite morphism
    :rtype: Nest_morphism
    """
    S = f.domain
    T = f.codomain
    U = g.domain
    V = g.codomain
    
    Tprime, Uprime = mutual_refinement(T, U)
    assert Tprime.refines(T) and Uprime.refines(U)
    
    inclusion = Nest_morphism(Tprime, Uprime, tuple(range(1, Tprime.length() + 1)))
    fprime = f.pullback_along(Tprime)
    gprime = g.pushforward_along(Uprime)
    
    return fprime.compose(inclusion).compose(gprime)


def main():
    """Main function for testing."""
    pass


if __name__ == "__main__":
    main()