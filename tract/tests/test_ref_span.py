"""
Test suite for the category Span(Tuple, Ref), accompanying Colfax
Research's work "Categorical Foundations for CuTe Layouts".

Tests validation, identity, composition (built on pullback along Ref
morphisms), sum, and the projection functor to the category Span (which
flattens the backward legs' nesting), verifying the category laws hold
strictly for the chosen pullback construction.

Run with: pytest tests/ref_span_tests.py
"""

import numpy as np
import pytest

from .conftest import seed_rngs

from tract import (
    NestedTuple,
    RefMorphism,
    RefSpanMorphism,
    SpanMorphism,
    TupleMorphism,
)

from .generators import random_Tuple_morphism

from .test_ref_morphism import random_nest_over

iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# RANDOM GENERATORS (local to this test suite for now)
# *************************************************************************


def random_RefSpan_morphism(domain=None) -> RefSpanMorphism:
    """
    Generate a random span, optionally with a prescribed domain U. The apex
    is a random nested refinement of U (built by factoring U's entries into
    randomly nested factorizations when U is prescribed).

    :param domain: Domain tuple U (optional)
    :type domain: Tuple[int] or None
    :return: Random span
    :rtype: RefSpanMorphism
    """
    if domain is None:
        right = random_Tuple_morphism(max_value=10)
        left = RefMorphism(NestedTuple(random_nest_over(right.domain)))
        return RefSpanMorphism(left, right)

    # Prescribed domain: refine each entry of U into a random nested
    # factorization to obtain the apex, then generate a random forward leg.
    top_data = []
    for t in domain:
        if t == 1:
            top_data.append(1)
            continue
        primes = []
        d, m = 2, t
        while m > 1:
            while m % d == 0:
                primes.append(d)
                m //= d
            d += 1
        np.random.shuffle(primes)
        factors = []
        current = 1
        for p in primes:
            current *= p
            if np.random.rand() < 0.5:
                factors.append(int(current))
                current = 1
        if current != 1 or not factors:
            factors.append(int(current))
        if len(factors) == 1:
            top_data.append(factors[0])
        else:
            top_data.append(random_nest_over(factors))
    left = RefMorphism(NestedTuple(tuple(top_data)))
    assert left.codomain == tuple(domain)
    right = random_Tuple_morphism(domain=left.domain, max_value=10)
    return RefSpanMorphism(left, right)


def random_composable_RefSpan_morphisms():
    """
    Generate a random composable pair (f, g) with f.codomain == g.domain.

    :return: Composable pair of spans
    :rtype: tuple[RefSpanMorphism, RefSpanMorphism]
    """
    f = random_RefSpan_morphism()
    g = random_RefSpan_morphism(domain=f.codomain)
    return f, g


# *************************************************************************
# TESTS
# *************************************************************************


class TestRefSpanMorphismValidation:
    """Tests for RefSpanMorphism construction and validation."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_random_construction_is_valid(self, iteration):
        """
        Test that randomly generated spans have consistent boundaries.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_RefSpan_morphism()
        assert f.apex == f.left.domain == f.right.domain
        assert f.domain == f.left.codomain
        assert f.codomain == f.right.codomain
        assert f.apex == f.left.nest.flatten()

    def test_mismatched_apex_raises(self):
        """
        Test that legs with different apexes are rejected.
        """
        left = RefMorphism(NestedTuple(((2, 3),)))
        right = TupleMorphism((5,), (5,), (1,))
        with pytest.raises(ValueError, match="apex"):
            RefSpanMorphism(left, right)

    def test_wrong_leg_types_raise(self):
        """
        Test that legs of the wrong type are rejected.
        """
        ref = RefMorphism(NestedTuple((6,)))
        tup = TupleMorphism((6,), (6,), (1,))
        with pytest.raises(ValueError, match="Left leg"):
            RefSpanMorphism(tup, tup)
        with pytest.raises(ValueError, match="Right leg"):
            RefSpanMorphism(ref, ref)


class TestRefSpanMorphismCategoryLaws:
    """Tests for identity, composition, and the category axioms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_identity(self, iteration):
        """
        Test that identity spans validate and report is_identity().

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_RefSpan_morphism()
        assert RefSpanMorphism.identity(f.domain).is_identity()

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_two_sided_unit(self, iteration):
        """
        Test that id ∘ f == f and f ∘ id == f hold strictly.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_RefSpan_morphism()
        assert f.compose(RefSpanMorphism.identity(f.codomain)) == f
        assert RefSpanMorphism.identity(f.domain).compose(f) == f

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_boundaries(self, iteration):
        """
        Test that composites have the correct domain and codomain.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.are_composable(g)
        composite = f.compose(g)
        assert composite.domain == f.domain
        assert composite.codomain == g.codomain
        assert composite.left.codomain == f.domain

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_not_composable_raises(self, iteration):
        """
        Test that composing non-composable spans raises ValueError.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_RefSpan_morphism()
        g = random_RefSpan_morphism()
        if f.codomain == g.domain:
            pytest.skip("Randomly generated spans happen to be composable")
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
            f, g = random_composable_RefSpan_morphisms()
            h = random_RefSpan_morphism(domain=g.codomain)
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.compose(g).compose(h) == f.compose(g.compose(h))


class TestRefSpanMorphismSum:
    """Tests for the legwise sum ⊕."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_concatenates(self, iteration):
        """
        Test that f ⊕ g concatenates apexes, domains, and codomains.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_RefSpan_morphism()
        g = random_RefSpan_morphism()
        s = f.sum(g)
        assert s.apex == f.apex + g.apex
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
            f, g = random_composable_RefSpan_morphisms()
            fprime, gprime = random_composable_RefSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.compose(g).sum(fprime.compose(gprime)) == f.sum(fprime).compose(
            g.sum(gprime)
        )


class TestRefSpanSpanBridge:
    """Tests for the projection functor Span(Tuple, Ref) → Span."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_to_Span_is_functorial(self, iteration):
        """
        Test that flattening the backward legs preserves composition.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        assert f.compose(g).to_span_morphism() == f.to_span_morphism().compose(
            g.to_span_morphism()
        )

    @pytest.mark.parametrize("iteration", iterations)
    def test_to_Span_preserves_identity_and_boundaries(self, iteration):
        """
        Test that the projection is the identity on objects and preserves
        identity spans.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_RefSpan_morphism()
        span = f.to_span_morphism()
        assert span.apex == f.apex
        assert span.domain == f.domain
        assert span.codomain == f.codomain
        assert RefSpanMorphism.identity(f.domain).to_span_morphism() == (
            SpanMorphism.identity(f.domain)
        )

    @pytest.mark.parametrize("iteration", iterations)
    def test_from_Span_roundtrip_and_composition(self, iteration):
        """
        Test that lifting Span morphisms is a section of the projection and
        that composites of lifted spans project back to Span composites.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_RefSpan_morphisms()
        except (ValueError, OverflowError) as e:
            pytest.skip(f"Skipped due to generator failure: {e}")
        span_f = f.to_span_morphism()
        span_g = g.to_span_morphism()
        lifted_f = RefSpanMorphism.from_span_morphism(span_f)
        lifted_g = RefSpanMorphism.from_span_morphism(span_g)
        assert lifted_f.to_span_morphism() == span_f
        assert lifted_g.to_span_morphism() == span_g
        assert lifted_f.compose(lifted_g).to_span_morphism() == span_f.compose(
            span_g
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
