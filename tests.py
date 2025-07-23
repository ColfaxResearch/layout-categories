import torch 
import numpy as np
from tqdm import tqdm
import cutlass
import cutlass.cute as cute 

from categories import *
from test_utils import *
from layout_utils import *

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)



#*************************************************************************
# INTERNAL TESTS
#*************************************************************************

def internal_sort_test(N: int):
    """
    Description: 
    Randomly generates N tuple morphisms, and checks that for each tuple morphism f, the tuple morphism sort(f) is sorted.
    """
    if N < 0: 
        raise ValueError("N must be a positive integer.")

    failure_count = 0
    for _ in range(N):
        f = random_Tuple_morphism()
        sort_f = f.sort()
        if not sort_f.is_sorted():
            failure_count += 1
    
    if failure_count == 0:
        print("sort(f) is sorted.")

    else:
        print("Sort test failed in",failure_count,"cases.")

def internal_coalesce_test(N: int):
    """
    Description: 
    Randomly generates N tuple morphisms, and checks that for each tuple morphism f, the tuple morphism coalesce(f) is coalesced.
    """
    if N < 0: 
        raise ValueError("N must be a positive integer.")

    failure_count = 0
    for _ in range(N):
        f = random_Tuple_morphism()
        coalesce_f = f.coalesce()
        if not coalesce_f.is_coalesced():
            failure_count += 1
    
    if failure_count == 0:
        print("coalesce(f) is coalesced.")

    else:
        print("Coalesce test failed in",failure_count,"cases.")



#*************************************************************************
# EXTERNAL TEST COMPONENTS
#*************************************************************************

@cute.jit
def test_coalesce_agree(f: cutlass.Constexpr[Tuple_morphism]):
    """
    Description: 
    Checks whether or not L_{coalesce(f)} is equal to coalesce(L_f).
    """
    coalesce_f = f.coalesce()
    coalesce_f.name = f"coalesce({f.name})"
    layout_f = compute_flat_layout(f)
    coalesce_layout = compute_flat_layout(coalesce_f)
    layout_coalesce = cute.coalesce(layout_f)
    if cute.rank(coalesce_layout) == 1:
        coalesce_layout = cute.make_layout(coalesce_layout.shape[0], stride=coalesce_layout.stride[0])
    
    # print(coalesce_layout)
    # print(layout_coalesce)

    agree = (coalesce_layout == layout_coalesce) or (cute.rank(coalesce_layout) == 0 and layout_coalesce == cute.make_layout(1,stride=0))
    return agree

@cute.jit 
def test_concat_agree(f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]):
    """
    Description: 
    Checks whether or not L_{concat(f,g)} is equal to concat(L_f,L_g).
    """
    layout_f = compute_flat_layout(f)
    layout_g = compute_flat_layout(g) 
    concat_morphs = f.concat(g)
    concat_morphs.name = f"concat({f.name}, {g.name})"
    layout_concat = compute_flat_layout(concat_morphs)
    concat_layout = flat_concatenate(layout_f, layout_g)
    agree = layout_concat == concat_layout
    return agree

@cute.jit
def test_compose_agree(f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]):    
    """
    Description: 
    Checks whether or not L_{g o f} is equal to L_f o L_f.
    """
    layout_f = compute_flat_layout(f)
    layout_g = compute_flat_layout(g)
    compose_morphs = f.compose(g)
    layout_compose = compute_flat_layout(compose_morphs)
    compose_layout = cute.composition(layout_g, layout_f)
    # print(compose_layout)
    # print(layout_compose)
    agree = layout_compose == compose_layout
    return agree



#*************************************************************************
# EXTERNAL TESTS
#*************************************************************************

def coalesce_test(N: int):
    """
    Description: 
    Randomly generates N tuple morphisms, and check whether or not L_{coalesce(f)} is equal to coalesce(L_f).
    """
    coalesce_agree = True
    for _ in range(10):
        f = random_Tuple_morphism()
        coalesce_agree = coalesce_agree and test_coalesce_agree(f)
    if coalesce_agree:
        print("L_{coalesce(f)} = coalesce(L_f)")
    else:
        print("Coalesce agreement test failed.")

def concat_test(N: int):
    """
    Description: 
    Randomly generates N pairs of tuple morphisms with disjoint images, and check whether or not L_{concat(f,g)} is equal to concat(L_f,L_g).
    """
    concat_agree = True
    for _ in range(N):
        f,g = random_Tuple_morphisms_with_disjoint_images()
        concat_agree = concat_agree and test_concat_agree(f,g)
    if concat_agree:
        print("L_concat(f,g) = concat(L_f,L_g)")
    else:
        print("Concatenation agreement test failed!")

def compose_test(N:int):
    """
    Description: 
    Randomly generates N pairs of composable tuple morphisms, and check whether or not L_{g o f} is equal to L_g o L_f.
    """
    compose_agree = True
    for _ in range(N):
        f,g = random_Tuple_composable_morphisms()   
        test_compose_agree(f,g)
        compose_agree = compose_agree and test_compose_agree(f,g)
    if compose_agree:
        print("L_{g o f} = L_g o L_f")
    else:
        print("Composition agreement test failed!")



#*************************************************************************
# MAIN
#*************************************************************************

def main():
    internal_sort_test(100)
    internal_coalesce_test(100)
    coalesce_test(1000)
    concat_test(100)
    compose_test(100)
    
if __name__ == "__main__":
    main()