"""
Tract: Categorical implementations of tractable CuTe layouts

A Python library accompanying Colfax Research's work "Categorical Foundations
for CuTe Layouts".

Example::

    from tract import TupleMorphism, compute_flat_layout

    f = TupleMorphism(domain=(4, 8, 2), codomain=(8, 2), map=(0, 1, 2))
    layout = compute_flat_layout(f)

For more examples, see the notebooks/ directory.
"""

__version__ = "0.1.0"
__author__ = "Colfax Research"

# Core categories
from .fin import FinMorphism
from .nested_tuple import NestedTuple
from .tuple_morphism import TupleMorphism
from .nest_morphism import (
    NestMorphism,
    make_morphism,
    compose,
    complement,
    coalesce,
    logical_divide,
    logical_product,
    morphism_to_tikz,
)

# The categories Fact and Ref
from .fact_morphism import FactMorphism
from .ref_morphism import RefMorphism

# Spans and cospans over Fact and Ref
from .spans import (
    SpanMorphism,
    CoSpanMorphism,
    RefSpanMorphism,
    RefCoSpanMorphism,
)

# Mutual refinement and weak composition (pure, no layout backend)
from .refinement import (
    mutual_refinement,
    weak_composite,
    mutual_refinement_to_tikz,
)

# Layout computation functions (CuTe DSL backend), loaded lazily so that
# importing tract does not require the cutlass DSL. The pycute backend is
# available as tract.backends.pycute.
_CUTE_BACKEND_NAMES = frozenset({
    "compute_flat_layout",
    "compute_flat_layout_components",
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
    "flat_complement",
    "layout_to_tikz",
})


def __getattr__(name):
    if name in _CUTE_BACKEND_NAMES:
        from .backends import cute_dsl

        return getattr(cute_dsl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Deprecated aliases (paper-era names)
Fin_morphism = FinMorphism
Tuple_morphism = TupleMorphism
Nest_morphism = NestMorphism
Fact_morphism = FactMorphism
Ref_morphism = RefMorphism
Span_morphism = SpanMorphism
CoSpan_morphism = CoSpanMorphism
RefSpan_morphism = RefSpanMorphism
RefCoSpan_morphism = RefCoSpanMorphism

__all__ = [
    "__version__",
    "__author__",

    # Core categories
    "FinMorphism",
    "NestedTuple",
    "TupleMorphism",
    "NestMorphism",
    "make_morphism",
    "compose",
    "complement",
    "coalesce",
    "logical_divide",
    "logical_product",
    "morphism_to_tikz",

    # Fact, Ref, and spans/cospans over them
    "FactMorphism",
    "RefMorphism",
    "SpanMorphism",
    "CoSpanMorphism",
    "RefSpanMorphism",
    "RefCoSpanMorphism",

    # Layout computation functions
    "compute_flat_layout",
    "compute_flat_layout_components",
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

    # Deprecated aliases
    "Fin_morphism",
    "Tuple_morphism",
    "Nest_morphism",
    "Fact_morphism",
    "Ref_morphism",
    "Span_morphism",
    "CoSpan_morphism",
    "RefSpan_morphism",
    "RefCoSpan_morphism",
]
