"""
Test suite for the category Ref, accompanying Colfax Research's work
"Categorical Foundations for CuTe Layouts".

Tests validation, identity, composition (grafting), sum, the pullback of
Tuple morphisms along Ref morphisms, and the functor Ref → Fact given by
flattening top-level modes.

Run with: pytest tests/ref_morphism_tests.py
"""

import numpy as np
import pytest

from .conftest import seed_rngs

from tract import (
    FactMorphism,
    NestedTuple,
    RefMorphism,
    TupleMorphism,
)

from .generators import random_Tuple_morphism

iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# RANDOM GENERATORS (local to this test suite for now)
# *************************************************************************


def random_nest_over(entries, group_prob=0.6, max_depth=3):
    """
    Build random nested tuple data whose flattening is `entries`, by
    recursively grouping consecutive entries.

    :param entries: Flat list of positive integers
    :type entries: list
    :param group_prob: Probability of nesting a block one level deeper
    :type group_prob: float
    :param max_depth: Maximum extra nesting depth
    :type max_depth: int
    :return: Nested tuple data (tuple of ints and tuples)
    :rtype: tuple
    """
    entries = list(entries)
    result = []
    while entries:
        k = int(np.random.randint(1, len(entries) + 1))
        block, entries = entries[:k], entries[k:]
        if len(block) > 1 and max_depth > 0 and np.random.rand() < group_prob:
            result.append(random_nest_over(block, group_prob, max_depth - 1))
        elif len(block) == 1:
            result.append(block[0])
        else:
            result.extend(block)
    return tuple(result)


def random_Ref_morphism(domain=None) -> RefMorphism:
    """
    Generate a random Ref morphism, optionally with prescribed domain (the
    flattening of its nested tuple).

    :param domain: Domain tuple (optional)
    :type domain: Tuple[int] or None
    :return: Random Ref morphism
    :rtype: RefMorphism
    """
    if domain is None:
        length = int(np.random.randint(0, 7))
        domain = tuple(int(x) for x in np.random.randint(1, 10, length))
    return RefMorphism(NestedTuple(random_nest_over(domain)))


def random_composable_Ref_morphisms():
    """
    Generate a random composable pair (f, g) with f.codomain == g.domain.

    :return: Composable pair of Ref morphisms
    :rtype: tuple[RefMorphism, RefMorphism]
    """
    f = random_Ref_morphism()
    g = random_Ref_morphism(domain=f.codomain)
    return f, g


# *************************************************************************
# TESTS
# *************************************************************************


class TestRefMorphismValidation:
    """Tests for RefMorphism construction and validation."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_random_construction_is_valid(self, iteration):
        """
        Test that the domain is the flattening and the codomain the depth-1
        reduction of the presenting nested tuple.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        assert f.domain == f.nest.flatten()
        assert f.codomain == tuple(
            f.nest.mode(i).size() for i in range(1, f.nest.rank() + 1)
        )
        assert f.size() == f.cosize()

    @pytest.mark.parametrize("iteration", iterations)
    def test_nest_refines_codomain(self, iteration):
        """
        Test that the presenting nested tuple refines the codomain.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        assert f.refined_codomain().refines(NestedTuple(f.codomain))

    def test_nonpositive_entries_raise(self):
        """
        Test that nonpositive entries are rejected.
        """
        with pytest.raises(ValueError, match="positive"):
            RefMorphism(NestedTuple((2, (0, 3))))
        with pytest.raises(ValueError, match="positive"):
            RefMorphism(NestedTuple((-1,)))

    def test_raw_data_is_wrapped(self):
        """
        Test that raw int/tuple data is accepted and wrapped.
        """
        f = RefMorphism(((2, 3), 4))
        assert f.domain == (2, 3, 4)
        assert f.codomain == (6, 4)

    def test_int_root_normalization(self):
        """
        Test that a bare-int nested tuple equals its singleton form.
        """
        assert RefMorphism(NestedTuple(5)) == RefMorphism(NestedTuple((5,)))


class TestRefMorphismCategoryLaws:
    """Tests for identity, composition, and the category axioms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_identity(self, iteration):
        """
        Test that identity morphisms validate and report is_identity().

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        identity = RefMorphism.identity(f.codomain)
        assert identity.is_identity()
        assert identity.domain == identity.codomain == f.codomain

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_two_sided_unit(self, iteration):
        """
        Test that id ∘ f == f and f ∘ id == f hold strictly.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        assert f.compose(RefMorphism.identity(f.codomain)) == f
        assert RefMorphism.identity(f.domain).compose(f) == f

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_domain_codomain(self, iteration):
        """
        Test that grafting produces the correct boundaries.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Ref_morphisms()
        composite = f.compose(g)
        assert composite.domain == f.domain
        assert composite.codomain == g.codomain

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_not_composable_raises(self, iteration):
        """
        Test that composing non-composable morphisms raises ValueError.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        g = random_Ref_morphism()
        if f.codomain == g.domain:
            pytest.skip("Randomly generated morphisms happen to be composable")
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
        f, g = random_composable_Ref_morphisms()
        h = random_Ref_morphism(domain=g.codomain)
        assert f.compose(g).compose(h) == f.compose(g.compose(h))


class TestRefMorphismSum:
    """Tests for the sum ⊕."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_concatenates(self, iteration):
        """
        Test that f ⊕ g concatenates domains and codomains.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        g = random_Ref_morphism()
        s = f.sum(g)
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
        f, g = random_composable_Ref_morphisms()
        fprime, gprime = random_composable_Ref_morphisms()
        assert f.compose(g).sum(fprime.compose(gprime)) == f.sum(fprime).compose(
            g.sum(gprime)
        )


class TestRefFactBridge:
    """Tests for the functor Ref → Fact and the Fact embedding."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_to_Fact_is_functorial(self, iteration):
        """
        Test that flattening top-level modes preserves composition.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Ref_morphisms()
        assert f.compose(g).to_fact_morphism() == f.to_fact_morphism().compose(
            g.to_fact_morphism()
        )

    @pytest.mark.parametrize("iteration", iterations)
    def test_to_Fact_preserves_identity_and_boundaries(self, iteration):
        """
        Test that the functor is the identity on objects and preserves
        identities.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        fact = f.to_fact_morphism()
        assert fact.domain == f.domain
        assert fact.codomain == f.codomain
        assert RefMorphism.identity(f.codomain).to_fact_morphism() == (
            FactMorphism.identity(f.codomain)
        )

    @pytest.mark.parametrize("iteration", iterations)
    def test_from_Fact_roundtrip(self, iteration):
        """
        Test that from_fact_morphism(f).to_fact_morphism() == f, and that
        the resulting Ref morphism has depth ≤ 2.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        fact = f.to_fact_morphism()
        lifted = RefMorphism.from_fact_morphism(fact)
        assert lifted.to_fact_morphism() == fact
        assert lifted.nest.depth() <= 2


class TestRefMorphismNestedTupleBridge:
    """Tests for from_refinement and refined_codomain."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_from_refinement_roundtrip(self, iteration):
        """
        Test that a Ref morphism can be rebuilt from the refinement
        nest ↠ codomain that it presents.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Ref_morphism()
        rebuilt = RefMorphism.from_refinement(
            f.refined_codomain(), NestedTuple(f.codomain)
        )
        assert rebuilt == f

    @pytest.mark.parametrize("iteration", iterations)
    def test_from_refinement_allows_nested_relative_modes(self, iteration):
        """
        Test that from_refinement accepts refinements with nested relative
        modes (which FactMorphism.from_refinement rejects).

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        refined = NestedTuple((((2, 3), 4), 5))
        coarse = NestedTuple((24, 5))
        f = RefMorphism.from_refinement(refined, coarse)
        assert f.domain == (2, 3, 4, 5)
        assert f.codomain == (24, 5)
        with pytest.raises(ValueError, match="not flat"):
            FactMorphism.from_refinement(refined, coarse)

    def test_from_refinement_rejects_non_flat_coarse(self):
        """
        Test that a non-flat coarse tuple is rejected.
        """
        with pytest.raises(ValueError, match="flat"):
            RefMorphism.from_refinement(
                NestedTuple(((2, 3), 4)), NestedTuple(((6,), 4))
            )


class TestRefMorphismPullback:
    """Tests for the pullback of Tuple morphisms along Ref morphisms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_pullback_agrees_with_Fact_pullback(self, iteration):
        """
        Test that the pullback delegates correctly to the underlying Fact
        morphism, and the induced refinement flattens to Fact's.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        b = random_Ref_morphism()
        f = random_Tuple_morphism(codomain=b.codomain, max_value=10)
        refinement, pulled = b.pullback_with_refinement(f)
        fact_refinement, fact_pulled = b.to_fact_morphism().pullback_with_refinement(f)
        assert pulled.domain == fact_pulled.domain
        assert pulled.codomain == fact_pulled.codomain
        assert pulled.map == fact_pulled.map
        assert refinement.to_fact_morphism() == fact_refinement

    @pytest.mark.parametrize("iteration", iterations)
    def test_pullback_boundaries(self, iteration):
        """
        Test the boundaries of the pullback square.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        b = random_Ref_morphism()
        f = random_Tuple_morphism(codomain=b.codomain, max_value=10)
        refinement, pulled = b.pullback_with_refinement(f)
        assert pulled.domain == refinement.domain
        assert pulled.codomain == b.domain
        assert refinement.codomain == f.domain

    @pytest.mark.parametrize("iteration", iterations)
    def test_pullback_along_identity_is_trivial(self, iteration):
        """
        Test that pulling back along an identity changes nothing.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        identity = RefMorphism.identity(f.codomain)
        refinement, pulled = identity.pullback_with_refinement(f)
        assert refinement.is_identity()
        assert pulled.domain == f.domain
        assert pulled.codomain == f.codomain
        assert pulled.map == f.map

    def test_mismatched_boundaries_raise(self):
        """
        Test that a codomain mismatch is rejected.
        """
        b = RefMorphism(NestedTuple(((2, 3),)))
        g = TupleMorphism((5,), (5,), (1,))
        with pytest.raises(ValueError, match="[Cc]odomain"):
            b.pullback_with_refinement(g)


class TestRefMorphismPushforward:
    """Tests for the pushforward of Tuple morphisms along Ref morphisms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_pushforward_agrees_with_Fact_pushforward(self, iteration):
        """
        Test that the pushforward delegates correctly to the underlying
        Fact morphism, and the induced refinement flattens to Fact's.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        b = random_Ref_morphism()
        f = random_Tuple_morphism(domain=b.codomain, max_value=10)
        refinement, pushed = b.pushforward_with_refinement(f)
        fact_refinement, fact_pushed = b.to_fact_morphism().pushforward_with_refinement(f)
        assert pushed.domain == fact_pushed.domain
        assert pushed.codomain == fact_pushed.codomain
        assert pushed.map == fact_pushed.map
        assert refinement.to_fact_morphism() == fact_refinement

    @pytest.mark.parametrize("iteration", iterations)
    def test_pushforward_boundaries(self, iteration):
        """
        Test the boundaries of the pushforward square.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        b = random_Ref_morphism()
        f = random_Tuple_morphism(domain=b.codomain, max_value=10)
        refinement, pushed = b.pushforward_with_refinement(f)
        assert pushed.domain == b.domain
        assert pushed.codomain == refinement.domain
        assert refinement.codomain == f.codomain

    @pytest.mark.parametrize("iteration", iterations)
    def test_pushforward_along_identity_is_trivial(self, iteration):
        """
        Test that pushing forward along an identity changes nothing.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        identity = RefMorphism.identity(f.domain)
        refinement, pushed = identity.pushforward_with_refinement(f)
        assert refinement.is_identity()
        assert pushed.domain == f.domain
        assert pushed.codomain == f.codomain
        assert pushed.map == f.map

    def test_pushforward_mismatched_boundaries_raise(self):
        """
        Test that a domain mismatch is rejected.
        """
        b = RefMorphism(NestedTuple(((2, 3),)))
        g = TupleMorphism((5,), (5,), (1,))
        with pytest.raises(ValueError, match="[Dd]omain"):
            b.pushforward_with_refinement(g)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
