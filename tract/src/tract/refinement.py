"""
Mutual refinement of nested tuples, and weak composition built on it.

Pure categorical operations: no layout backend is involved.
"""

from .nested_tuple import NestedTuple
from .nest_morphism import NestMorphism


def mutual_refinement(nestedtuple1: NestedTuple, nestedtuple2: NestedTuple):
    """
    Compute the mutual refinement of two nested tuples.

    Given nested tuples T and U, computes T' and U' such that T' refines T,
    U' refines U, and T' divides U'. Example::

        T = (6,6), U = (2,6,6)  ->  T' = ((2,3),(2,3)), U' = (2,(3,2),(3,2))

    :raises ValueError: If the tuples are not mutually refinable
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


def weak_composite(f: NestMorphism, g: NestMorphism) -> NestMorphism:
    """
    Compute the weak composition of nested tuple morphisms: composition
    through the mutual refinement of f's codomain with g's domain.
    """
    T = f.codomain
    U = g.domain

    Tprime, Uprime = mutual_refinement(T, U)
    assert Tprime.refines(T) and Uprime.refines(U)

    inclusion = NestMorphism(Tprime, Uprime, tuple(range(1, Tprime.length() + 1)))
    fprime = f.pullback_along(Tprime)
    gprime = g.pushforward_along(Uprime)

    return fprime.compose(inclusion).compose(gprime)


def mutual_refinement_to_tikz(
    nestedtuple1: NestedTuple, nestedtuple2: NestedTuple
) -> str:
    """Generate a TikZ diagram of the mutual refinement of two nested tuples."""
    from .tuple_morph_tikz import two_parenthesizations_to_tikz_values

    Tprime, Uprime = mutual_refinement(nestedtuple1, nestedtuple2)
    return two_parenthesizations_to_tikz_values(
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
