"""
Test suite for the category CoSpan, accompanying Colfax Research's work
"Categorical Foundations for CuTe Layouts".

Tests validation, identity, composition (built on pushforward along Fact
morphisms), and sum for CoSpanMorphism, verifying the category laws hold
strictly for the chosen pushforward construction.

Run with: pytest tests/cospan_tests.py
"""

import numpy as np
import pytest

from .conftest import seed_rngs

from tract import (
    CoSpanMorphism,
    FactMorphism,
    TupleMorphism,
)

from .generators import random_Tuple_morphism

iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# RANDOM GENERATORS (local to this test suite for now)
# *************************************************************************


def random_Fact_coarsening_of(domain) -> FactMorphism:
    """
    Generate a random Fact morphism with the given domain, by grouping the
    domain into random consecutive blocks.

    :param domain: Domain tuple
    :type domain: Tuple[int]
    :return: Random Fact morphism out of domain
    :rtype: FactMorphism
    """
    entries = list(domain)
    modes = []
    while entries:
        k = np.random.randint(1, len(entries) + 1)
        modes.append(tuple(entries[:k]))
        entries = entries[k:]
    modes = tuple(modes)
    codomain = tuple(int(np.prod(mode)) for mode in modes)
    return FactMorphism(tuple(domain), codomain, modes)


def random_CoSpan_morphism(domain=None) -> CoSpanMorphism:
    """
    Generate a random cospan, optionally with a prescribed domain U. The
    forward leg is a random Tuple morphism out of U, and the backward leg is
    a random coarsening of its codomain (the nadir).

    :param domain: Domain tuple U (optional)
    :type domain: Tuple[int] or None
    :return: Random cospan
    :rtype: CoSpanMorphism
    """
    if domain is None:
        left = random_Tuple_morphism(max_value=10)
    else:
        left = random_Tuple_morphism(domain=tuple(domain), max_value=10)
    right = random_Fact_coarsening_of(left.codomain)
    return CoSpanMorphism(left, right)


def random_composable_CoSpan_morphisms():
    """
    Generate a random composable pair (f, g) with f.codomain == g.domain.
    Raises ValueError if the chained cosizes overflow the generator's
    2³⁰ - 1 ceiling; tests skip in that case, following the OverflowError
    convention of morphism_tests.py.

    :return: Composable pair of cospans
    :rtype: tuple[CoSpanMorphism, CoSpanMorphism]
    """
    f = random_CoSpan_morphism()
    g = random_CoSpan_morphism(domain=f.codomain)
    return f, g


# *************************************************************************
# TESTS
# *************************************************************************


class TestCoSpanMorphismValidation:
    """Tests for CoSpanMorphism construction and validation."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_random_construction_is_valid(self, iteration):
        """
        Test that randomly generated cospans have consistent boundaries.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_CoSpan_morphism()
        assert f.nadir == f.left.codomain == f.right.domain
        assert f.domain == f.left.domain
        assert f.codomain == f.right.codomain

    def test_mismatched_nadir_raises(self):
        """
        Test that legs with different nadirs are rejected.
        """
        left = TupleMorphism((5,), (5,), (1,))
        right = FactMorphism((2, 3), (6,), ((2, 3),))
        with pytest.raises(ValueError, match="nadir"):
            CoSpanMorphism(left, right)

    def test_wrong_leg_types_raise(self):
        """
        Test that legs of the wrong type are rejected.
        """
        fact = FactMorphism((6,), (6,), ((6,),))
        tup = TupleMorphism((6,), (6,), (1,))
        with pytest.raises(ValueError, match="Left leg"):
            CoSpanMorphism(fact, fact)
        with pytest.raises(ValueError, match="Right leg"):
            CoSpanMorphism(tup, tup)


class TestCoSpanMorphismCategoryLaws:
    """Tests for identity, composition, and the category axioms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_identity(self, iteration):
        """
        Test that identity cospans validate and report is_identity().

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_CoSpan_morphism()
        assert CoSpanMorphism.identity(f.domain).is_identity()

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_two_sided_unit(self, iteration):
        """
        Test that id ∘ f == f and f ∘ id == f hold strictly.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_CoSpan_morphism()
        assert f.compose(CoSpanMorphism.identity(f.codomain)) == f
        assert CoSpanMorphism.identity(f.domain).compose(f) == f

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_boundaries(self, iteration):
        """
        Test that composites have the correct domain and codomain, and that
        the composite nadir refines the second nadir onward to g's codomain.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_CoSpan_morphisms()
        except ValueError as e:
            pytest.skip(f"Skipped due to cosize overflow in generator: {e}")
        assert f.are_composable(g)
        composite = f.compose(g)
        assert composite.domain == f.domain
        assert composite.codomain == g.codomain
        assert composite.left.domain == f.domain

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_not_composable_raises(self, iteration):
        """
        Test that composing non-composable cospans raises ValueError.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_CoSpan_morphism()
        g = random_CoSpan_morphism()
        if f.codomain == g.domain:
            pytest.skip("Randomly generated cospans happen to be composable")
        with pytest.raises(ValueError, match="not composable"):
            f.compose(g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_composition_is_associative(self, iteration):
        """
        Test that (h ∘ g) ∘ f == h ∘ (g ∘ f) holds strictly on random
        composable triples.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_CoSpan_morphisms()
            h = random_CoSpan_morphism(domain=g.codomain)
        except ValueError as e:
            pytest.skip(f"Skipped due to cosize overflow in generator: {e}")
        assert f.compose(g).compose(h) == f.compose(g.compose(h))


class TestCoSpanMorphismSum:
    """Tests for the legwise sum ⊕."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_concatenates(self, iteration):
        """
        Test that f ⊕ g concatenates nadirs, domains, and codomains.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_CoSpan_morphism()
        g = random_CoSpan_morphism()
        s = f.sum(g)
        assert s.nadir == f.nadir + g.nadir
        assert s.domain == f.domain + g.domain
        assert s.codomain == f.codomain + g.codomain

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_compose_interchange(self, iteration):
        """
        Test the interchange law (g ∘ f) ⊕ (g' ∘ f') == (g ⊕ g') ∘ (f ⊕ f').

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_CoSpan_morphisms()
            fprime, gprime = random_composable_CoSpan_morphisms()
        except ValueError as e:
            pytest.skip(f"Skipped due to cosize overflow in generator: {e}")
        assert f.compose(g).sum(fprime.compose(gprime)) == f.sum(fprime).compose(
            g.sum(gprime)
        )
