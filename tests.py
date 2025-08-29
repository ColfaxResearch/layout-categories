import numpy as np
import pytest
import cutlass
import cutlass.cute as cute 

from categories import *
from test_utils import *
from layout_utils import *

iterations = range(1000)

#*************************************************************************
# TUPLE_MORPHISM TEST COMPONENTS
#*************************************************************************

@cute.jit
def coalesce_agree(f: cutlass.Constexpr[Tuple_morphism]):
    coalesce_f      = f.coalesce()
    layout_f        = compute_flat_layout(f)
    coalesce_layout = compute_flat_layout(coalesce_f)
    layout_coalesce = cute.coalesce(layout_f)
    agree = (coalesce_layout == layout_coalesce) or (cute.rank(coalesce_layout) == 0 and layout_coalesce == cute.make_layout(1,stride=0)) or (cute.rank(coalesce_layout) == 1 and layout_coalesce == cute.make_layout(coalesce_layout.shape[0], stride=coalesce_layout.stride[0]))
    return agree

@cute.jit 
def concat_agree(f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]):
    layout_f      = compute_flat_layout(f)
    layout_g      = compute_flat_layout(g) 
    concat_morphs = f.concat(g)
    layout_concat = compute_flat_layout(concat_morphs)
    concat_layout = flat_concatenate(layout_f, layout_g)
    return layout_concat == concat_layout

@cute.jit
def compose_agree(f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]):    
    layout_f       = compute_flat_layout(f)
    layout_g       = compute_flat_layout(g)
    compose_morphs = f.compose(g)
    layout_compose = nullify_trivial_strides(compute_flat_layout(compose_morphs))
    compose_layout = cute.composition(layout_g, layout_f)
    return layout_compose == compose_layout

@cute.jit
def complement_agree(f: cutlass.Constexpr[Tuple_morphism]):
    f_complement        = f.complement()
    layout_f            = compute_flat_layout(f)
    layout_f_complement = compute_flat_layout(f_complement)
    complement_layout_f = cute.complement(layout_f,f.cosize())
    return cute.coalesce(complement_layout_f) == cute.coalesce(layout_f_complement)

@cute.jit
def flat_divide_agree(f: cutlass.Constexpr[Tuple_morphism],g: cutlass.Constexpr[Tuple_morphism]):
    layout_f        = compute_flat_layout(f)
    layout_g        = compute_flat_layout(g)
    quotient        = f.flat_divide(g)
    # if cute.rank(layout_g) == 0:
    #     layout_g = cute.make_layout(1,stride = 0)
    quotient_layout = flatten_layout(cute.logical_divide(layout_f,layout_g))
    layout_quotient = compute_flat_layout(quotient)
    return cute.coalesce(layout_quotient) == cute.coalesce(quotient_layout)

@cute.jit
def flat_product_agree(f:cutlass.Constexpr[Tuple_morphism],g:cutlass.Constexpr[Tuple_morphism]):
    k = f.flat_product(g)
    A = compute_flat_layout(f)
    B = compute_flat_layout(g)
    C = compute_flat_layout(k)
    product = flatten_layout(cute.logical_product(A,B))
    return nullify_trivial_strides(C) == nullify_trivial_strides(product)

#*************************************************************************
# TUPLE_MORPHISM TESTS
#*************************************************************************

@pytest.mark.parametrize("iteration", iterations)
def test_sort_is_sorted(iteration):
    """
    Description: 
    If f is a tuple morphism, then sort(f) is sorted.
    """
    np.random.seed(iteration)
    f = random_Tuple_morphism()
    assert f.sort().is_sorted()

@pytest.mark.parametrize("iteration", iterations)
def test_coalesce_is_coalesced(iteration):
    """
    Description: 
    If f is a tuple morphism, then coalesce(f) is coalesced.
    """
    np.random.seed(iteration)
    f = random_Tuple_morphism()
    assert f.coalesce().is_coalesced()

@pytest.mark.parametrize("iteration", iterations)
def test_complement_is_a_complement(iteration):
    """
    Description: 
    If f is a complementable tuple morphism, then complement(f) is a complement of f.
    """
    np.random.seed(iteration)
    f = random_Tuple_complementable_morphism()
    assert f.is_complementary_to(f.complement())

@pytest.mark.parametrize("iteration",iterations)
def test_coalesce_agree(iteration):
    """
    Description: 
    If f is a tuple morphsm, then L_{coalesce(f)} = coalesce(L_f)
    if we consider ():() = 1:0, and (s):(d) = s:d
    """
    np.random.seed(iteration)
    f = random_Tuple_morphism()
    assert coalesce_agree(f)

@pytest.mark.parametrize("iteration",iterations)
def test_concat_agree(iteration):
    """
    Description: 
    If f and g are tuple morphisms with the same codomain and disjoint images,
    then L_{concat(f,g)} = concat(L_f,L_g).
    """
    f,g = random_Tuple_morphisms_with_disjoint_images()
    assert concat_agree(f,g)

@pytest.mark.parametrize("iteration",iterations)
def test_compose_agree(iteration):
    """
    Description: 
    If f and g are composable tuple morphisms, then L_{g o f} = L_g o L_f
    (after nullifying trivial strides)
    """
    f,g = random_Tuple_composable_morphisms()
    assert compose_agree(f,g)

@pytest.mark.parametrize("iteration",iterations)
def test_complement_agree(iteration):
    """
    Description: 
    If f is a complementable tuple morphisms, then then 
    coalesce(L_{complement(f)}) = coalesce(complement(L_f))
    """
    np.random.seed(iteration)
    f = random_Tuple_complementable_morphism()
    assert complement_agree(f)

@pytest.mark.parametrize("iteration",iterations)
def test_flat_divide_agree(iteration):
    """
    Description: 
    If g divides f, then 
    coalesce(L_{f/g}) = coalesce(flatten(L_f oslash L_g))
    """
    np.random.seed(iteration)
    f,g = random_Tuple_divisible_morphisms()
    assert flat_divide_agree(f,g)

@pytest.mark.parametrize("iteration",iterations)
def test_flat_product_agree(iteration):
    """
    Description: 
    If f and g are productable tuple morphisms, then 
    L_{f x g} = flatten(L_f otimes L_g) (after nullifying trivial strides)
    """
    np.random.seed(iteration)
    f,g = random_Tuple_productable_morphisms()
    assert flat_product_agree(f,g)


#*************************************************************************
# NEST_MORPHISM TEST COMPONENTS
#*************************************************************************

@cute.jit 
def Nest_concat_agree(f: cutlass.Constexpr[Nest_morphism], g: cutlass.Constexpr[Nest_morphism]):
    layout_f      = compute_layout(f)
    layout_g      = compute_layout(g) 
    concat_morphs = f.concat(g)
    layout_concat = compute_layout(concat_morphs)
    concat_layout = concatenate(layout_f, layout_g)
    return layout_concat == concat_layout

@cute.jit
def Nest_compose_agree(f: cutlass.Constexpr[Nest_morphism], g: cutlass.Constexpr[Nest_morphism]):    
    layout_f       = compute_layout(f)
    layout_g       = compute_layout(g)
    compose_morphs = f.compose(g)
    layout_compose = nullify_trivial_strides(compute_flat_layout(compose_morphs))
    compose_layout = cute.composition(layout_g, layout_f)
    return layout_compose == compose_layout


#*************************************************************************
# NEST_MORPHISM TESTS
#*************************************************************************

@pytest.mark.parametrize("iteration", iterations)
def test_Nest_complement_is_a_complement(iteration):
    """
    Description: 
    If f is a complementable tuple morphism, then complement(f) is a complement of f.
    """
    np.random.seed(iteration)
    f = random_Nest_complementable_morphism()
    assert f.is_complementary_to(f.complement())

@pytest.mark.parametrize("iteration", iterations)
def test_Nest_concat_agree(iteration):
    np.random.seed(iteration)
    f,g = random_Nest_morphisms_with_disjoint_images()
    assert Nest_concat_agree(f,g)


