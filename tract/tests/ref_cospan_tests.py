"""
Test suite for the category CoSpan(Tuple, Ref), accompanying Colfax
Research's work "Categorical Foundations for CuTe Layouts".

Tests validation, identity, composition (built on pushforward along Ref
morphisms), sum, and the projection functor to the category CoSpan (which
flattens the backward legs' nesting), verifying the category laws hold
strictly for the chosen pushforward construction.

Run with: pytest tests/ref_cospan_tests.py
"""

import numpy as np
import pytest

from tract import (
    CoSpan_morphism,
    NestedTuple,
    Ref_morphism,
    RefCoSpan_morphism,
    Tuple_morphism,
    random_Tuple_morphism,
)

from .ref_morphism_tests import random_nest_over

iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# RANDOM GENERATORS (local to this test suite for now)
# *************************************************************************


def random_RefCoSpan_morphism(domain=None) -> RefCoSpan_morphism:
    """
    Generate a random cospan, optionally with a prescribed domain U. The
    forward leg is a random Tuple morphism out of U, and the backward leg
    is a random nested coarsening of its codomain (the nadir).

    :param domain: Domain tuple U (optional)
    :type domain: Tuple[int] or None
    :return: Random cospan
    :rtype: RefCoSpan_morphism
    """
    if domain is None:
        left = random_Tuple_morphism(max_value=10)
    else:
        left = random_Tuple_morphism(domain=tuple(domain), max_value=10)
    right = Ref_morphism(NestedTuple(random_nest_over(left.codomain)))
    return RefCoSpan_morphism(left, right)


def random_composable_RefCoSpan_morphisms():
    """
    Generate a random composable pair (f, g) with f.codomain == g.domain.
    Raises ValueError if the chained cosizes overflow the generator's
    2³⁰ - 1 ceiling; tests skip in that case, following the OverflowError
    convention of morphism_tests.py.

    :return: Composable pair of cospans
    :rtype: tuple[RefCoSpan_morphism, RefCoSpan_morphism]
    """
    f = random_RefCoSpan_morphism()
    g = random_RefCoSpan_morphism(domain=f.codomain)
    return f, g


# *************************************************************************
# TESTS
# *************************************************************************


class TestRefCoSpanMorphismValidation:
    """Tests for RefCoSpan_morphism construction and validation."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_random_construction_is_valid(self, iteration):
        """
        Test that randomly generated cospans have consistent boundaries.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_RefCoSpan_morphism()
        assert f.nadir == f.left.codomain == f.right.domain
        assert f.domain == f.left.domain
        assert f.codomain == f.right.codomain
        assert f.nadir == f.right.nest.flatten()

    def test_mismatched_nadir_raises(self):
        """
        Test that legs with different nadirs are rejected.
        """
        left = Tuple_morphism((5,), (5,), (1,))
        right = Ref_morphism(NestedTuple(((2, 3),)))
        with pytest.raises(ValueError, match="nadir"):
            RefCoSpan_morphism(left, right)

    def test_wrong_leg_types_raise(self):
        """
        Test that legs of the wrong type are rejected.
        """
        ref = Ref_morphism(NestedTuple((6,)))
        tup = Tuple_morphism((6,), (6,), (1,))
        with pytest.raises(ValueError, match="Left leg"):
            RefCoSpan_morphism(ref, ref)
        with pytest.raises(ValueError, match="Right leg"):
            RefCoSpan_morphism(tup, tup)


class TestRefCoSpanMorphismCategoryLaws:
    """Tests for identity, composition, and the category axioms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_identity(self, iteration):
        """
        Test that identity cospans validate and report is_identity().

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_RefCoSpan_morphism()
        assert RefCoSpan_morphism.identity(f.domain).is_identity()

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_two_sided_unit(self, iteration):
        """
        Test that id ∘ f == f and f ∘ id == f hold strictly.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_RefCoSpan_morphism()
        assert f.compose(RefCoSpan_morphism.identity(f.codomain)) == f
        assert RefCoSpan_morphism.identity(f.domain).compose(f) == f

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_boundaries(self, iteration):
        """
        Test that composites have the correct domain and codomain.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefCoSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.are_composable(g)
        composite = f.compose(g)
        assert composite.domain == f.domain
        assert composite.codomain == g.codomain
        assert composite.right.codomain == g.codomain

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_not_composable_raises(self, iteration):
        """
        Test that composing non-composable cospans raises ValueError.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_RefCoSpan_morphism()
        g = random_RefCoSpan_morphism()
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
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefCoSpan_morphisms()
            h = random_RefCoSpan_morphism(domain=g.codomain)
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.compose(g).compose(h) == f.compose(g.compose(h))


class TestRefCoSpanMorphismSum:
    """Tests for the legwise sum ⊕."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_concatenates(self, iteration):
        """
        Test that f ⊕ g concatenates nadirs, domains, and codomains.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_RefCoSpan_morphism()
        g = random_RefCoSpan_morphism()
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
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefCoSpan_morphisms()
            fprime, gprime = random_composable_RefCoSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.compose(g).sum(fprime.compose(gprime)) == f.sum(fprime).compose(
            g.sum(gprime)
        )


class TestRefCoSpanCoSpanBridge:
    """Tests for the projection functor CoSpan(Tuple, Ref) → CoSpan."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_to_CoSpan_is_functorial(self, iteration):
        """
        Test that flattening the backward legs preserves composition.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefCoSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.compose(g).to_CoSpan_morphism() == f.to_CoSpan_morphism().compose(
            g.to_CoSpan_morphism()
        )

    @pytest.mark.parametrize("iteration", iterations)
    def test_to_CoSpan_preserves_identity_and_boundaries(self, iteration):
        """
        Test that the projection is the identity on objects and preserves
        identity cospans.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_RefCoSpan_morphism()
        cospan = f.to_CoSpan_morphism()
        assert cospan.nadir == f.nadir
        assert cospan.domain == f.domain
        assert cospan.codomain == f.codomain
        assert RefCoSpan_morphism.identity(f.domain).to_CoSpan_morphism() == (
            CoSpan_morphism.identity(f.domain)
        )

    @pytest.mark.parametrize("iteration", iterations)
    def test_from_CoSpan_roundtrip_and_composition(self, iteration):
        """
        Test that lifting CoSpan morphisms is a section of the projection
        and that composites of lifted cospans project back to CoSpan
        composites.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefCoSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        cospan_f = f.to_CoSpan_morphism()
        cospan_g = g.to_CoSpan_morphism()
        lifted_f = RefCoSpan_morphism.from_CoSpan_morphism(cospan_f)
        lifted_g = RefCoSpan_morphism.from_CoSpan_morphism(cospan_g)
        assert lifted_f.to_CoSpan_morphism() == cospan_f
        assert lifted_g.to_CoSpan_morphism() == cospan_g
        assert lifted_f.compose(lifted_g).to_CoSpan_morphism() == cospan_f.compose(
            cospan_g
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
