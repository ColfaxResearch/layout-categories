# ============================================================================
# src/tract/__init__.py
# ============================================================================

"""
Tract: Categorical implementations of tractable CuTe layouts

A Python library accompanying Colfax Research's work "Categorical Foundations for CuTe Layouts".

Examples
--------
Basic usage::

    from tract import Tuple_morphism, compute_flat_layout
    
    # Create a simple tuple morphism
    f = Tuple_morphism(
        domain=(4, 8, 2),
        codomain=(8, 2),
        map=(0, 1, 2)
    )
    
    # Compute associated layout
    layout = compute_flat_layout(f)

For more examples, see the examples/ directory.
"""

__version__ = "0.1.0"
__author__ = "Colfax Research"

# Core morphism classes
from .categories import (
    Fin_morphism,
    Tuple_morphism,
    Nest_morphism,
    NestedTuple,
    make_morphism,
    compose,
    complement,
    coalesce,
    logical_divide,
    logical_product,
    morphism_to_tikz,
)

# Layout computation functions
from .layout_utils import (
    compute_flat_layout,
    compute_layout,
    compute_Tuple_morphism,
    compute_Nest_morphism,
    compute_morphism,
    flatten_layout,
    sort_flat_layout,
    sort_flat_layout_with_perm,
    is_tractable,
    flat_concatenate,
    concatenate,
    nullify_trivial_strides,
    nullify_zero_strides,
    mutual_refinement,
    weak_composite,
    flat_complement,
    layout_to_tikz,
    mutual_refinement_to_tikz,
)

from .test_utils import (
    random_Fin_morphism,
    random_Tuple_morphism,
    random_Nest_morphism,
    random_NestedTuple,
    random_complementable_Fin_morphism,
    random_complementable_Tuple_morphism,
    random_complementable_Nest_morphism,
    random_composable_Tuple_morphisms,
    random_composable_Nest_morphisms,
    random_divisible_Tuple_morphisms,
    random_divisible_Nest_morphisms,
    random_mutually_refinable_nested_tuples,
    random_ordered_subtuple,
    random_profile,
    random_product_admissible_Tuple_morphisms,
    random_product_admissible_Nest_morphisms,
    random_Tuple_morphisms_with_disjoint_images,
    random_Nest_morphisms_with_disjoint_images,
    random_weakly_composable_nest_morphisms,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Core morphism classes
    "Fin_morphism",
    "Tuple_morphism", 
    "Nest_morphism",
    "NestedTuple",
    "make_morphism",
    "compose",
    "complement",
    "coalesce",
    "logical_divide",
    "logical_product",
    "morphism_to_tikz",
    
    # Layout computation functions
    "compute_flat_layout",
    "compute_layout",
    "compute_Tuple_morphism",
    "compute_Nest_morphism",
    "compute_morphism",
    "flatten_layout",
    "sort_flat_layout",
    "sort_flat_layout_with_perm",
    "is_tractable",
    "flat_concatenate",
    "concatenate",
    "nullify_trivial_strides",
    "nullify_zero_strides",
    "mutual_refinement",
    "weak_composite",
    "flat_complement",
    "layout_to_tikz",
    "mutual_refinement_to_tikz",
    
    # Test utilities
    "random_Fin_morphism",
    "random_Tuple_morphism",
    "random_Nest_morphism",
    "random_NestedTuple",
    "random_complementable_Fin_morphism",
    "random_complementable_Tuple_morphism",
    "random_complementable_Nest_morphism",
    "random_composable_Tuple_morphisms",
    "random_composable_Nest_morphisms",
    "random_divisible_Tuple_morphisms",
    "random_divisible_Nest_morphisms",
    "random_mutually_refinable_nested_tuples",
    "random_ordered_subtuple",
    "random_profile",
    "random_product_admissible_Tuple_morphisms",
    "random_product_admissible_Nest_morphisms",
    "random_Tuple_morphisms_with_disjoint_images",
    "random_Nest_morphisms_with_disjoint_images",
    "random_weakly_composable_nest_morphisms",
]
