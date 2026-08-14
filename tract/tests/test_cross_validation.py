"""
Cross-validation of tract's categorical operations against the CuTe layout
algebra, run against every available backend (the cutlass CuTe DSL and
NVIDIA's pycute reference implementation). Each agreement predicate is
written once against the adapter interface in backend_adapters.py.
"""

import pytest

from tract import NestMorphism, TupleMorphism
from tract.refinement import mutual_refinement

from .conftest import seed_rngs
from .generators import (
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

iterations = range(100)
RANDOM_SEED_BASE = 42


@pytest.fixture(scope="module", params=["cute", "pycute"])
def bk(request):
    """One adapter per available layout backend; skip if not installed."""
    if request.param == "cute":
        pytest.importorskip("cutlass")
        from .backend_adapters import CuteBackend

        return CuteBackend()
    pytest.importorskip("pycute")
    from .backend_adapters import PycuteBackend

    return PycuteBackend()


# *************************************************************************
# AGREEMENT PREDICATES
# *************************************************************************


def coalesce_agree(bk, f: TupleMorphism) -> bool:
    """Morphism coalescence matches layout coalescence."""
    coalesce_f = f.coalesce()
    layout_f = bk.compute_flat_layout(f)
    coalesce_layout = bk.compute_flat_layout(coalesce_f)
    layout_coalesce = bk.coalesce(layout_f)
    return bk.layouts_agree(coalesce_layout, layout_coalesce)


def concat_agree(bk, f: TupleMorphism, g: TupleMorphism) -> bool:
    """Morphism concatenation matches flat layout concatenation."""
    layout_f = bk.compute_flat_layout(f)
    layout_g = bk.compute_flat_layout(g)
    layout_concat = bk.compute_flat_layout(f.concat(g))
    concat_layout = bk.flat_concatenate(layout_f, layout_g)
    return layout_concat == concat_layout


def compose_agree(bk, f: TupleMorphism, g: TupleMorphism) -> bool:
    """Morphism composition matches layout composition."""
    layout_f = bk.compute_flat_layout(f)
    layout_g = bk.compute_flat_layout(g)
    layout_compose = bk.compute_flat_layout(f.compose(g))
    compose_layout = bk.composition(layout_g, layout_f)
    return bk.layouts_agree(layout_compose, compose_layout)


def complement_agree(bk, f: TupleMorphism) -> bool:
    """Morphism complement matches layout complement, up to coalescence."""
    layout_f = bk.compute_flat_layout(f)
    layout_f_complement = bk.compute_flat_layout(f.complement())
    complement_layout_f = bk.complement(layout_f, f.cosize())
    return bk.layouts_agree(
        bk.coalesce(complement_layout_f), bk.coalesce(layout_f_complement)
    )


def flat_divide_agree(bk, f: TupleMorphism, g: TupleMorphism) -> bool:
    """Flat division of morphisms matches logical division of layouts."""
    layout_f = bk.compute_flat_layout(f)
    layout_g = bk.compute_flat_layout(g)
    quotient_layout = bk.flatten_layout(bk.logical_divide(layout_f, layout_g))
    layout_quotient = bk.compute_flat_layout(f.flat_divide(g))
    return bk.layouts_agree(bk.coalesce(layout_quotient), bk.coalesce(quotient_layout))


def flat_product_agree(bk, f: TupleMorphism, g: TupleMorphism) -> bool:
    """Flat product of morphisms matches logical product of layouts."""
    k = f.flat_product(g)
    A = bk.compute_flat_layout(f)
    B = bk.compute_flat_layout(g)
    C = bk.compute_flat_layout(k)
    product = bk.flatten_layout(bk.logical_product(A, B))
    return C == product


def Nest_concat_agree(bk, f: NestMorphism, g: NestMorphism) -> bool:
    """Nested morphism concatenation matches nested layout concatenation."""
    layout_f = bk.compute_layout(f)
    layout_g = bk.compute_layout(g)
    layout_concat = bk.compute_layout(f.concat(g))
    concat_layout = bk.concatenate(layout_f, layout_g)
    return layout_concat == concat_layout


def Nest_complement_agree(bk, f: NestMorphism) -> bool:
    """Nested morphism complement matches layout complement."""
    layout_f = bk.compute_layout(f)
    layout_f_complement = bk.compute_layout(f.complement())
    complement_layout_f = bk.complement(layout_f, f.cosize())
    return bk.layouts_agree(complement_layout_f, bk.coalesce(layout_f_complement))


def Nest_compose_agree(bk, f: NestMorphism, g: NestMorphism) -> bool:
    """Nested morphism composition matches layout composition."""
    layout_f = bk.compute_layout(f)
    layout_g = bk.compute_layout(g)
    layout_compose = bk.compute_layout(f.compose(g))
    compose_layout = bk.composition(layout_g, layout_f)
    return bk.layouts_agree(layout_compose, compose_layout)


def Nest_coalesce_agree(bk, f: NestMorphism) -> bool:
    """Nested morphism coalescence matches layout coalescence."""
    layout_f = bk.compute_layout(f)
    coalesce_layout = bk.compute_layout(f.coalesce())
    layout_coalesce = bk.coalesce(layout_f)
    return bk.layouts_agree(coalesce_layout, layout_coalesce)


def Nest_logical_product_agree(bk, f: NestMorphism, g: NestMorphism) -> bool:
    """Nested morphism logical product matches layout logical product."""
    layout_f = bk.compute_layout(f)
    layout_g = bk.compute_layout(g)
    product_layout = bk.logical_product(layout_f, layout_g)
    layout_product = bk.compute_layout(f.logical_product(g))
    return layout_product == product_layout


def Nest_logical_divide_agree(bk, f: NestMorphism, g: NestMorphism) -> bool:
    """Nested morphism logical division matches layout logical division."""
    layout_quotient = bk.compute_layout(f.logical_divide(g))
    layout_f = bk.compute_layout(f)
    layout_g = bk.compute_layout(g)
    layout_g_complement = bk.complement(layout_g, bk.size(layout_f))
    quotient_layout = bk.composition(
        layout_f, bk.concatenate(layout_g, layout_g_complement)
    )
    return bk.layouts_agree(bk.coalesce(layout_quotient), bk.coalesce(quotient_layout))


def composition_algorithm_agree(bk, f: NestMorphism, g: NestMorphism) -> bool:
    """The weak composition algorithm matches direct layout composition."""
    S = f.domain
    T = f.codomain
    U = g.domain

    Tprime, Uprime = mutual_refinement(T, U)
    fprime = f.pullback_along(Tprime)
    inclusion = NestMorphism(Tprime, Uprime, tuple(range(1, Tprime.length() + 1)))
    gprime = g.pushforward_along(Uprime)
    weak_composite = bk.compute_layout(fprime.compose(inclusion).compose(gprime))
    composite = bk.coalesce_to_profile(weak_composite, S.data)
    return bk.layouts_agree(
        composite, bk.composition(bk.compute_layout(g), bk.compute_layout(f))
    )


def flat_layout_round_trip_agree(bk, f: TupleMorphism) -> bool:
    """L_f is tractable and L → f_L → L_{f_L} recovers it."""
    layout_f = bk.compute_flat_layout(f)
    if not bk.is_tractable(layout_f):
        return False
    round_trip = bk.compute_flat_layout(bk.compute_Tuple_morphism(layout_f))
    return bk.layouts_agree(round_trip, layout_f)


def nest_layout_round_trip_agree(bk, f: NestMorphism) -> bool:
    """L_f is tractable and L → f_L → L_{f_L} recovers it."""
    layout_f = bk.compute_layout(f)
    if not bk.is_tractable(layout_f):
        return False
    round_trip = bk.compute_layout(bk.compute_Nest_morphism(layout_f))
    return bk.layouts_agree(round_trip, layout_f)


# *************************************************************************
# TESTS
# *************************************************************************


class TestTupleMorphismAgreement:
    """Cross-validation of TupleMorphism operations against CuTe layouts."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_coalesce_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        assert bk.check(coalesce_agree, f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_concat_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_Tuple_morphisms_with_disjoint_images()
            assert bk.check(concat_agree, f, g)
        except OverflowError as e:
            pytest.skip(f"Skipped due to cosize overflow: {e}")

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Tuple_morphisms()
        assert bk.check(compose_agree, f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_complement_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Tuple_morphism(max_value=10)
        assert bk.check(complement_agree, f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_flat_divide_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_divisible_Tuple_morphisms()
        assert bk.check(flat_divide_agree, f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_flat_product_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_product_admissible_Tuple_morphisms()
        assert bk.check(flat_product_agree, f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_layout_morphism_round_trip(self, bk, iteration):
        """L → f_L → L_{f_L} recovers a tractable flat layout."""
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        assert bk.check(flat_layout_round_trip_agree, f)


class TestNestMorphismAgreement:
    """Cross-validation of NestMorphism operations against CuTe layouts."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_complement_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Nest_morphism(max_value=10)
        assert bk.check(Nest_complement_agree, f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_concat_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_Nest_morphisms_with_disjoint_images()
            assert bk.check(Nest_concat_agree, f, g)
        except OverflowError as e:
            pytest.skip(f"Skipped due to cosize overflow: {e}")

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_coalesce_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Nest_morphism(max_value=10)
        assert bk.check(Nest_coalesce_agree, f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_logical_divide_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_divisible_Nest_morphisms()
        assert bk.check(Nest_logical_divide_agree, f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_logical_product_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_product_admissible_Nest_morphisms()
        assert bk.check(Nest_logical_product_agree, f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_compose_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Nest_morphisms(
            min_length=0, max_length=6, max_value=64
        )
        assert bk.check(Nest_compose_agree, f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_composition_algorithm_agree(self, bk, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        T, U = random_mutually_refinable_nested_tuples()
        f = random_Nest_morphism(codomain=T, max_length=8, max_value=16)
        g = random_Nest_morphism(domain=U, max_length=8, max_value=16)
        assert bk.check(composition_algorithm_agree, f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_layout_morphism_round_trip(self, bk, iteration):
        """L → f_L → L_{f_L} recovers a tractable nested layout."""
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Nest_morphism(max_value=10)
        assert bk.check(nest_layout_round_trip_agree, f)
