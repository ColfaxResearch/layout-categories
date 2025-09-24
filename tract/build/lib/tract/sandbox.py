import numpy as np
import pytest
import cutlass
import cutlass.cute as cute


# Import from the tract package
from tract.categories import (
    Fin_morphism,
    Tuple_morphism,
    Nest_morphism,
    NestedTuple
)

from tract.test_utils import (
    random_Tuple_morphism,
    random_complementable_Tuple_morphism,
    random_composable_Tuple_morphisms,
    random_Tuple_morphisms_with_disjoint_images,
    random_divisible_Tuple_morphisms,
    random_product_admissible_Tuple_morphisms,
    random_complementable_Nest_morphism,
    random_Nest_morphisms_with_disjoint_images,
    random_composable_Nest_morphisms,
    random_product_admissible_Nest_morphisms,
    random_divisible_Nest_morphisms,
    random_mutually_refinable_nested_tuples,
    random_Nest_morphism,
)

from tract.layout_utils import (
    compute_flat_layout,
    compute_layout,
    flatten_layout,
    flat_concatenate,
    concatenate,
    nullify_trivial_strides,
    nullify_zero_strides,
    mutual_refinement)


@cute.jit
def example():
    A = cute.make_layout((),stride = ())
    print(A)
    print(cute.coalesce(A))

for _ in range(20):
    f,g = random_divisible_Nest_morphisms(max_value = 3)
    print(f)
    print(g)
    print("-"*40)

