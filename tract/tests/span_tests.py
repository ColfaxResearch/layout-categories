"""
Test suite for the category Span, accompanying Colfax Research's work
"Categorical Foundations for CuTe Layouts".

Tests validation, identity, composition (built on pullback along Fact
morphisms), and sum for Span_morphism, verifying the category laws hold
strictly for the chosen pullback construction.

Run with: pytest tests/span_tests.py
"""

import numpy as np
import pytest

from tract import Fact_morphism, Span_morphism, Tuple_morphism, random_Tuple_morphism

iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# RANDOM GENERATORS (local to this test suite for now)
# *************************************************************************


def random_Fact_coarsening_of(domain) -> Fact_morphism:
    """
    Generate a random Fact morphism with the given domain, by grouping the
    domain into random consecutive blocks.

    :param domain: Domain tuple
    :type domain: Tuple[int]
    :return: Random Fact morphism out of domain
    :rtype: Fact_morphism
    """
    entries = list(domain)
    modes = []
    while entries:
        k = np.random.randint(1, len(entries) + 1)
        modes.append(tuple(entries[:k]))
        entries = entries[k:]
    modes = tuple(modes)
    codomain = tuple(int(np.prod(mode)) for mode in modes)
    return Fact_morphism(tuple(domain), codomain, modes)


def random_Span_morphism(domain=None) -> Span_morphism:
    """
    Generate a random span, optionally with a prescribed domain U. The apex
    is a random refinement of U (built by factoring U's entries via a random
    Tuple morphism's domain when U is not prescribed).

    :param domain: Domain tuple U (optional)
    :type domain: Tuple[int] or None
    :return: Random span
    :rtype: Span_morphism
    """
    if domain is None:
        right = random_Tuple_morphism(max_value=10)
        left = random_Fact_coarsening_of(right.domain)
        return Span_morphism(left, right)

    # Prescribed domain: refine each entry of U into a random factorization
    # to obtain the apex, then generate a random forward leg out of it.
    modes = []
    for t in domain:
        if t == 1:
            modes.append((1,))
            continue
        primes = []
        d, m = 2, t
        while m > 1:
            while m % d == 0:
                primes.append(d)
                m //= d
            d += 1
        np.random.shuffle(primes)
        mode = []
        current = 1
        for p in primes:
            current *= p
            if np.random.rand() < 0.5:
                mode.append(int(current))
                current = 1
        if current != 1 or not mode:
            mode.append(int(current))
        modes.append(tuple(mode))
    modes = tuple(modes)
    apex = tuple(entry for mode in modes for entry in mode)
    left = Fact_morphism(apex, tuple(domain), modes)
    right = random_Tuple_morphism(domain=apex, max_value=10)
    return Span_morphism(left, right)


def random_composable_Span_morphisms():
    """
    Generate a random composable pair (f, g) with f.codomain == g.domain.

    :return: Composable pair of spans
    :rtype: tuple[Span_morphism, Span_morphism]
    """
    f = random_Span_morphism()
    g = random_Span_morphism(domain=f.codomain)
    return f, g


# *************************************************************************
# TESTS
# *************************************************************************


class TestSpanMorphismValidation:
    """Tests for Span_morphism construction and validation."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_random_construction_is_valid(self, iteration):
        """
        Test that randomly generated spans have consistent boundaries.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Span_morphism()
        assert f.apex == f.left.domain == f.right.domain
        assert f.domain == f.left.codomain
        assert f.codomain == f.right.codomain

    def test_mismatched_apex_raises(self):
        """
        Test that legs with different apexes are rejected.
        """
        left = Fact_morphism((2, 3), (6,), ((2, 3),))
        right = Tuple_morphism((5,), (5,), (1,))
        with pytest.raises(ValueError, match="apex"):
            Span_morphism(left, right)

    def test_wrong_leg_types_raise(self):
        """
        Test that legs of the wrong type are rejected.
        """
        fact = Fact_morphism((6,), (6,), ((6,),))
        tup = Tuple_morphism((6,), (6,), (1,))
        with pytest.raises(ValueError, match="Left leg"):
            Span_morphism(tup, tup)
        with pytest.raises(ValueError, match="Right leg"):
            Span_morphism(fact, fact)


class TestSpanMorphismCategoryLaws:
    """Tests for identity, composition, and the category axioms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_identity(self, iteration):
        """
        Test that identity spans validate and report is_identity().

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Span_morphism()
        assert Span_morphism.identity(f.domain).is_identity()

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_two_sided_unit(self, iteration):
        """
        Test that id ∘ f == f and f ∘ id == f hold strictly.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Span_morphism()
        assert f.compose(Span_morphism.identity(f.codomain)) == f
        assert Span_morphism.identity(f.domain).compose(f) == f

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_boundaries(self, iteration):
        """
        Test that composites have the correct domain and codomain, and that
        the composite apex refines the first apex (via the left leg of the
        pullback square).

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_Span_morphisms()
        except ValueError as e:
            pytest.skip(f"Skipped due to cosize overflow in generator: {e}")
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
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Span_morphism()
        g = random_Span_morphism()
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
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_Span_morphisms()
            h = random_Span_morphism(domain=g.codomain)
        except ValueError as e:
            pytest.skip(f"Skipped due to cosize overflow in generator: {e}")
        assert f.compose(g).compose(h) == f.compose(g.compose(h))


class TestSpanMorphismSum:
    """Tests for the legwise sum ⊕."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_concatenates(self, iteration):
        """
        Test that f ⊕ g concatenates apexes, domains, and codomains.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Span_morphism()
        g = random_Span_morphism()
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
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_composable_Span_morphisms()
            fprime, gprime = random_composable_Span_morphisms()
        except ValueError as e:
            pytest.skip(f"Skipped due to cosize overflow in generator: {e}")
        assert f.compose(g).sum(fprime.compose(gprime)) == f.sum(fprime).compose(
            g.sum(gprime)
        )
