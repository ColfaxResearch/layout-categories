"""
Cross-validation test suite for tract against pycute.

Mirrors the layout-agreement tests in morphism_tests.py, but validates
tract's categorical operations against NVIDIA's pycute
(https://github.com/NVlabs/CuTe), the official pure-Python reference
implementation of CuTe. No GPU or cutlass DSL is required.

pycute is not a PyPI dependency of tract; install it from a local clone,
e.g. ``uv pip install -e /path/to/CuTe``.
"""

import numpy as np
import pytest

pycute = pytest.importorskip("pycute")

from pycute import (
    coalesce,
    complement,
    logical_divide,
    size,
)

from tract.categories import (
    Tuple_morphism,
    Nest_morphism,
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

from tract.pycute_utils import (
    compute_flat_layout,
    compute_layout,
    compute_Tuple_morphism,
    compute_Nest_morphism,
    flatten_layout,
    flat_concatenate,
    concatenate,
    layouts_agree,
    is_tractable,
    compose_layouts,
    coalesce_layout,
    logical_product_layouts,
)

from tract.layout_utils import mutual_refinement

# Test configuration
iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# TUPLE_MORPHISM TEST COMPONENTS
# *************************************************************************


def coalesce_agree(f: Tuple_morphism) -> bool:
    """
    Check if morphism coalescence agrees with pycute layout coalescence.

    :param f: Tuple morphism to test
    :type f: Tuple_morphism
    :return: True if coalescence operations agree
    :rtype: bool
    """
    coalesce_f = f.coalesce()
    layout_f = compute_flat_layout(f)
    coalesce_layout = compute_flat_layout(coalesce_f)
    layout_coalesce = coalesce(layout_f)
    return layouts_agree(coalesce_layout, layout_coalesce)


def concat_agree(f: Tuple_morphism, g: Tuple_morphism) -> bool:
    """
    Check if morphism concatenation agrees with layout concatenation.

    :param f: First morphism
    :type f: Tuple_morphism
    :param g: Second morphism
    :type g: Tuple_morphism
    :return: True if concatenations agree
    :rtype: bool
    """
    layout_f = compute_flat_layout(f)
    layout_g = compute_flat_layout(g)
    concat_morphs = f.concat(g)
    layout_concat = compute_flat_layout(concat_morphs)
    concat_layout = flat_concatenate(layout_f, layout_g)
    return layout_concat == concat_layout


def compose_agree(f: Tuple_morphism, g: Tuple_morphism) -> bool:
    """
    Check if morphism composition agrees with pycute layout composition.

    :param f: First morphism
    :type f: Tuple_morphism
    :param g: Second morphism
    :type g: Tuple_morphism
    :return: True if compositions agree
    :rtype: bool
    """
    layout_f = compute_flat_layout(f)
    layout_g = compute_flat_layout(g)
    compose_morphs = f.compose(g)
    layout_compose = compute_flat_layout(compose_morphs)
    compose_layout = compose_layouts(layout_g, layout_f)
    return layouts_agree(layout_compose, compose_layout)


def complement_agree(f: Tuple_morphism) -> bool:
    """
    Check if morphism complement agrees with pycute layout complement.

    :param f: Morphism to test
    :type f: Tuple_morphism
    :return: True if complements agree
    :rtype: bool
    """
    f_complement = f.complement()
    layout_f = compute_flat_layout(f)
    layout_f_complement = compute_flat_layout(f_complement)
    complement_layout_f = complement(layout_f, f.cosize())
    return layouts_agree(
        coalesce(complement_layout_f), coalesce(layout_f_complement)
    )


def flat_divide_agree(f: Tuple_morphism, g: Tuple_morphism) -> bool:
    """
    Check if flat division of morphisms agrees with pycute layout division.

    :param f: Numerator morphism
    :type f: Tuple_morphism
    :param g: Denominator morphism
    :type g: Tuple_morphism
    :return: True if divisions agree
    :rtype: bool
    """
    layout_f = compute_flat_layout(f)
    layout_g = compute_flat_layout(g)
    quotient = f.flat_divide(g)
    quotient_layout = flatten_layout(logical_divide(layout_f, layout_g))
    layout_quotient = compute_flat_layout(quotient)
    return layouts_agree(coalesce(layout_quotient), coalesce(quotient_layout))


def flat_product_agree(f: Tuple_morphism, g: Tuple_morphism) -> bool:
    """
    Check if flat product of morphisms agrees with pycute layout product.

    :param f: First factor morphism
    :type f: Tuple_morphism
    :param g: Second factor morphism
    :type g: Tuple_morphism
    :return: True if products agree
    :rtype: bool
    """
    k = f.flat_product(g)
    A = compute_flat_layout(f)
    B = compute_flat_layout(g)
    C = compute_flat_layout(k)
    product = flatten_layout(logical_product_layouts(A, B))
    return C == product


# *************************************************************************
# TUPLE_MORPHISM TESTS
# *************************************************************************


class TestTupleMorphismAgainstPycute:
    """Cross-validation of Tuple_morphism operations against pycute."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_coalesce_agree(self, iteration):
        """
        Test that morphism coalescence agrees with pycute coalescence.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        assert coalesce_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_concat_agree(self, iteration):
        """
        Test that morphism concatenation agrees with layout concatenation.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_Tuple_morphisms_with_disjoint_images()
        assert concat_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_agree(self, iteration):
        """
        Test that morphism composition agrees with pycute composition.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Tuple_morphisms()
        assert compose_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_complement_agree(self, iteration):
        """
        Test that morphism complement agrees with pycute complement.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Tuple_morphism(max_value=10)
        assert complement_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_flat_divide_agree(self, iteration):
        """
        Test that flat division agrees between morphisms and pycute layouts.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_divisible_Tuple_morphisms()
        assert flat_divide_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_flat_product_agree(self, iteration):
        """
        Test that flat product agrees between morphisms and pycute layouts.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_product_admissible_Tuple_morphisms()
        assert flat_product_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_layout_morphism_round_trip(self, iteration):
        """
        Test that L → f_L → L_{f_L} recovers a tractable flat layout.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        layout_f = compute_flat_layout(f)
        assert is_tractable(layout_f)
        round_trip = compute_flat_layout(compute_Tuple_morphism(layout_f))
        assert layouts_agree(round_trip, layout_f)


# *************************************************************************
# NEST_MORPHISM TEST COMPONENTS
# *************************************************************************


def Nest_concat_agree(f: Nest_morphism, g: Nest_morphism) -> bool:
    """
    Check if nested morphism concatenation agrees with layout concatenation.

    :param f: First morphism
    :type f: Nest_morphism
    :param g: Second morphism
    :type g: Nest_morphism
    :return: True if concatenations agree
    :rtype: bool
    """
    layout_f = compute_layout(f)
    layout_g = compute_layout(g)
    concat_morphs = f.concat(g)
    layout_concat = compute_layout(concat_morphs)
    concat_layout = concatenate(layout_f, layout_g)
    return layout_concat == concat_layout


def Nest_complement_agree(f: Nest_morphism) -> bool:
    """
    Check if nested morphism complement agrees with pycute complement.

    :param f: Morphism to test
    :type f: Nest_morphism
    :return: True if complements agree
    :rtype: bool
    """
    f_complement = f.complement()
    layout_f = compute_layout(f)
    layout_f_complement = compute_layout(f_complement)
    complement_layout_f = complement(layout_f, f.cosize())
    return layouts_agree(complement_layout_f, coalesce(layout_f_complement))


def Nest_compose_agree(f: Nest_morphism, g: Nest_morphism) -> bool:
    """
    Check if nested morphism composition agrees with pycute composition.

    :param f: First morphism
    :type f: Nest_morphism
    :param g: Second morphism
    :type g: Nest_morphism
    :return: True if compositions agree
    :rtype: bool
    """
    layout_f = compute_layout(f)
    layout_g = compute_layout(g)
    compose_morphs = f.compose(g)
    layout_compose = compute_layout(compose_morphs)
    compose_layout = compose_layouts(layout_g, layout_f)
    return layouts_agree(layout_compose, compose_layout)


def Nest_coalesce_agree(f: Nest_morphism) -> bool:
    """
    Check if nested morphism coalescence agrees with pycute coalescence.

    :param f: Nest morphism to test
    :type f: Nest_morphism
    :return: True if coalesce operations agree
    :rtype: bool
    """
    coalesce_f = f.coalesce()
    layout_f = compute_layout(f)
    coalesce_layout = compute_layout(coalesce_f)
    layout_coalesce = coalesce(layout_f)
    return layouts_agree(coalesce_layout, layout_coalesce)


def Nest_logical_product_agree(f: Nest_morphism, g: Nest_morphism) -> bool:
    """
    Check if nested morphism logical product agrees with pycute.
    """
    layout_f = compute_layout(f)
    layout_g = compute_layout(g)
    product = f.logical_product(g)
    product_layout = logical_product_layouts(layout_f, layout_g)
    layout_product = compute_layout(product)
    return layout_product == product_layout


def Nest_logical_divide_agree(f: Nest_morphism, g: Nest_morphism) -> bool:
    """
    Check if nested morphism logical division agrees with pycute.
    """
    morphism_quotient = f.logical_divide(g)
    layout_quotient = compute_layout(morphism_quotient)
    layout_f = compute_layout(f)
    layout_g = compute_layout(g)
    layout_g_complement = complement(layout_g, size(layout_f))
    quotient_layout = compose_layouts(
        layout_f, concatenate(layout_g, layout_g_complement)
    )
    return layouts_agree(coalesce(layout_quotient), coalesce(quotient_layout))


def composition_algorithm_agree(f: Nest_morphism, g: Nest_morphism) -> bool:
    """
    Check if the weak composition algorithm agrees with pycute composition.

    :param f: First morphism
    :type f: Nest_morphism
    :param g: Second morphism
    :type g: Nest_morphism
    :return: True if algorithm is correct
    :rtype: bool
    """
    S = f.domain
    T = f.codomain
    U = g.domain

    Tprime, Uprime = mutual_refinement(T, U)
    fprime = f.pullback_along(Tprime)
    inclusion = Nest_morphism(
        Tprime, Uprime, tuple(range(1, Tprime.length() + 1))
    )
    gprime = g.pushforward_along(Uprime)
    weak_composite = compute_layout(fprime.compose(inclusion).compose(gprime))
    composite = coalesce_layout(weak_composite, S.data)
    return layouts_agree(
        composite, compose_layouts(compute_layout(g), compute_layout(f))
    )


# *************************************************************************
# NEST_MORPHISM TESTS
# *************************************************************************


class TestNestMorphismAgainstPycute:
    """Cross-validation of Nest_morphism operations against pycute."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_complement_agree(self, iteration):
        """
        Test that nested morphism complement agrees with pycute complement.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Nest_morphism(max_value=10)
        assert Nest_complement_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_concat_agree(self, iteration):
        """
        Test that nested morphism concatenation agrees with layout.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_Nest_morphisms_with_disjoint_images()
        assert Nest_concat_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_coalesce_agree(self, iteration):
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Nest_morphism(max_value=10)
        assert Nest_coalesce_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_logical_divide_agree(self, iteration):
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_divisible_Nest_morphisms()
        assert Nest_logical_divide_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_logical_product_agree(self, iteration):
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_product_admissible_Nest_morphisms()
        assert Nest_logical_product_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_compose_agree(self, iteration):
        """
        Test that nested morphism composition agrees with pycute.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Nest_morphisms(
            min_length=0, max_length=6, max_value=64
        )
        assert Nest_compose_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_composition_algorithm_agree(self, iteration):
        """
        Test that the weak composition algorithm agrees with pycute.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        T, U = random_mutually_refinable_nested_tuples()
        f = random_Nest_morphism(codomain=T, max_length=8, max_value=16)
        g = random_Nest_morphism(domain=U, max_length=8, max_value=16)
        assert composition_algorithm_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_layout_morphism_round_trip(self, iteration):
        """
        Test that L → f_L → L_{f_L} recovers a tractable nested layout.
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Nest_morphism(max_value=10)
        layout_f = compute_layout(f)
        assert is_tractable(layout_f)
        round_trip = compute_layout(compute_Nest_morphism(layout_f))
        assert round_trip == layout_f


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
