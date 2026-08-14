"""
Test suite for the category Fact, accompanying Colfax Research's work
"Categorical Foundations for CuTe Layouts".

Tests validation, identity, composition, sum, and the NestedTuple bridge for
Fact_morphism. Unlike morphism_tests.py, these tests have no CuTe counterpart
to compare against, so they verify the category laws directly using the
structural equality on Fact_morphism.

Run with: pytest tests/fact_morphism_tests.py
"""

import numpy as np
import pytest

from tract import Fact_morphism, NestedTuple, random_Tuple_morphism

iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# RANDOM GENERATORS (local to this test suite for now)
# *************************************************************************


def random_Fact_morphism(
    max_codomain_length: int = 4, max_mode_length: int = 3, max_entry: int = 8
) -> Fact_morphism:
    """
    Generate a random Fact morphism by generating random modes and deriving
    the domain (flattening) and codomain (entrywise products).

    :param max_codomain_length: Maximum codomain length
    :type max_codomain_length: int
    :param max_mode_length: Maximum length of each mode
    :type max_mode_length: int
    :param max_entry: Maximum value of each domain entry
    :type max_entry: int
    :return: Random Fact morphism
    :rtype: Fact_morphism
    """
    n = np.random.randint(1, max_codomain_length + 1)
    modes = []
    for _ in range(n):
        k = np.random.randint(1, max_mode_length + 1)
        modes.append(tuple(int(x) for x in np.random.randint(1, max_entry + 1, k)))
    modes = tuple(modes)
    domain = tuple(entry for mode in modes for entry in mode)
    codomain = tuple(int(np.prod(mode)) for mode in modes)
    return Fact_morphism(domain, codomain, modes)


def random_coarsening(f: Fact_morphism) -> Fact_morphism:
    """
    Generate a random Fact morphism whose domain is f's codomain, by grouping
    the codomain of f into consecutive blocks.

    :param f: Fact morphism to coarsen the codomain of
    :type f: Fact_morphism
    :return: Random Fact morphism composable with f
    :rtype: Fact_morphism
    """
    entries = list(f.codomain)
    modes = []
    while entries:
        k = np.random.randint(1, len(entries) + 1)
        modes.append(tuple(entries[:k]))
        entries = entries[k:]
    modes = tuple(modes)
    codomain = tuple(int(np.prod(mode)) for mode in modes)
    return Fact_morphism(f.codomain, codomain, modes)


def random_Fact_refinement_of(codomain) -> Fact_morphism:
    """
    Generate a random Fact morphism with the given codomain, by choosing a
    random factorization of each codomain entry (a random consecutive
    grouping of a random ordering of its prime factors).

    :param codomain: Codomain tuple
    :type codomain: Tuple[int]
    :return: Random Fact morphism onto codomain
    :rtype: Fact_morphism
    """
    modes = []
    for t in codomain:
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
    domain = tuple(entry for mode in modes for entry in mode)
    return Fact_morphism(domain, tuple(codomain), modes)


def random_composable_Fact_morphisms():
    """
    Generate a random composable pair (f, g) with f.codomain == g.domain.

    :return: Composable pair of Fact morphisms
    :rtype: tuple[Fact_morphism, Fact_morphism]
    """
    f = random_Fact_morphism()
    g = random_coarsening(f)
    return f, g


# *************************************************************************
# TESTS
# *************************************************************************


class TestFactMorphismValidation:
    """Tests for Fact_morphism construction and validation."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_random_construction_is_valid(self, iteration):
        """
        Test that randomly generated morphisms pass validation and have
        consistent data.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        assert len(f.modes) == len(f.codomain)
        assert tuple(entry for mode in f.modes for entry in mode) == f.domain

    def test_invalid_product_raises(self):
        """
        Test that a mode whose product differs from the codomain entry is
        rejected.
        """
        with pytest.raises(ValueError, match="prod"):
            Fact_morphism((2, 3), (5,), ((2, 3),))

    def test_invalid_flattening_raises(self):
        """
        Test that modes whose flattening differs from the domain are rejected.
        """
        with pytest.raises(ValueError, match="Flattening"):
            Fact_morphism((3, 2), (6,), ((2, 3),))

    def test_wrong_mode_count_raises(self):
        """
        Test that a mode count different from the codomain length is rejected.
        """
        with pytest.raises(ValueError, match="Number of modes"):
            Fact_morphism((2, 3), (2, 3), ((2, 3),))

    def test_nonpositive_entries_raise(self):
        """
        Test that non-positive entries in domain, codomain, or modes are
        rejected.
        """
        with pytest.raises(ValueError, match="positive"):
            Fact_morphism((0, 3), (0, 3), ((0,), (3,)))
        with pytest.raises(ValueError, match="positive"):
            Fact_morphism((2,), (-2,), ((2,),))

    def test_empty_mode_is_allowed(self):
        """
        Test that an empty mode (empty product = 1) over a codomain entry of 1
        is a valid morphism.
        """
        f = Fact_morphism((2,), (2, 1), ((2,), ()))
        assert f.size() == f.cosize() == 2

    @pytest.mark.parametrize("iteration", iterations)
    def test_size_equals_cosize(self, iteration):
        """
        Test that Fact morphisms preserve size.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        assert f.size() == f.cosize()


class TestFactMorphismCategoryLaws:
    """Tests for identity, composition, and the category axioms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_identity(self, iteration):
        """
        Test that identity morphisms validate and report is_identity().

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        assert Fact_morphism.identity(f.codomain).is_identity()
        assert f.is_identity() == all(len(mode) == 1 for mode in f.modes)

    @pytest.mark.parametrize("iteration", iterations)
    def test_identity_is_two_sided_unit(self, iteration):
        """
        Test that id ∘ f == f and f ∘ id == f.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        assert f.compose(Fact_morphism.identity(f.codomain)) == f
        assert Fact_morphism.identity(f.domain).compose(f) == f

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_domain_codomain(self, iteration):
        """
        Test that composites have the correct domain and codomain (validity is
        checked by the Fact_morphism constructor during composition).

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Fact_morphisms()
        assert f.are_composable(g)
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
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        g = random_Fact_morphism()
        if f.codomain == g.domain:
            pytest.skip("Randomly generated morphisms happen to be composable")
        with pytest.raises(ValueError, match="not composable"):
            f.compose(g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_composition_is_associative(self, iteration):
        """
        Test that (h ∘ g) ∘ f == h ∘ (g ∘ f) on random composable triples.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Fact_morphisms()
        h = random_coarsening(g)
        assert f.compose(g).compose(h) == f.compose(g.compose(h))


class TestFactMorphismSum:
    """Tests for the monoidal sum ⊕."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_concatenates(self, iteration):
        """
        Test that f ⊕ g concatenates domains, codomains, and modes.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        g = random_Fact_morphism()
        s = f.sum(g)
        assert s.domain == f.domain + g.domain
        assert s.codomain == f.codomain + g.codomain
        assert s.modes == f.modes + g.modes

    @pytest.mark.parametrize("iteration", iterations)
    def test_sum_compose_interchange(self, iteration):
        """
        Test the interchange law (g ∘ f) ⊕ (g' ∘ f') == (g ⊕ g') ∘ (f ⊕ f').

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Fact_morphisms()
        fprime, gprime = random_composable_Fact_morphisms()
        assert f.compose(g).sum(fprime.compose(gprime)) == f.sum(fprime).compose(
            g.sum(gprime)
        )


class TestFactMorphismPullbackPushforward:
    """Tests for pullback and pushforward of Tuple morphisms along Fact morphisms."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_pullback_agrees_with_refinement_version(self, iteration):
        """
        Test that pulling back a Tuple morphism along a Fact morphism agrees
        with the general refinement implementation
        (Nest_morphism.pullback_along).

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        F = random_Fact_refinement_of(f.codomain)
        special = F.pullback(f)
        general = f.to_Nest_morphism().pullback_along(F.refined_codomain())
        assert special.domain == general.domain.flatten()
        assert special.codomain == general.codomain.flatten()
        assert special.map == general.map

    @pytest.mark.parametrize("iteration", iterations)
    def test_pushforward_agrees_with_refinement_version(self, iteration):
        """
        Test that pushing forward a Tuple morphism along a Fact morphism
        agrees with the general refinement implementation
        (Nest_morphism.pushforward_along).

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        F = random_Fact_refinement_of(f.domain)
        special = F.pushforward(f)
        general = f.to_Nest_morphism().pushforward_along(F.refined_codomain())
        assert special.domain == general.domain.flatten()
        assert special.codomain == general.codomain.flatten()
        assert special.map == general.map

    @pytest.mark.parametrize("iteration", iterations)
    def test_pullback_boundaries(self, iteration):
        """
        Test that the pullback lands where it should: codomain is the Fact
        morphism's domain, and domain entries over the image are refined.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        F = random_Fact_refinement_of(f.codomain)
        pullback = F.pullback(f)
        assert pullback.codomain == F.domain
        assert pullback.size() % f.size() == 0

    @pytest.mark.parametrize("iteration", iterations)
    def test_pullback_along_identity_is_trivial(self, iteration):
        """
        Test that pulling back and pushing forward along an identity Fact
        morphism does nothing.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        for g in (
            Fact_morphism.identity(f.codomain).pullback(f),
            Fact_morphism.identity(f.domain).pushforward(f),
        ):
            assert (g.domain, g.codomain, g.map) == (f.domain, f.codomain, f.map)

    def test_mismatched_boundaries_raise(self):
        """
        Test that pullback and pushforward reject mismatched morphisms.
        """
        F = Fact_morphism((2, 3), (6,), ((2, 3),))
        f = random_Tuple_morphism(max_value=10)
        if f.codomain != F.codomain:
            with pytest.raises(ValueError, match="Codomain"):
                F.pullback(f)
        if f.domain != F.codomain:
            with pytest.raises(ValueError, match="Domain"):
                F.pushforward(f)


class TestFactMorphismNestedTupleBridge:
    """Tests relating Fact morphisms to NestedTuple refinements."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_refined_codomain_refines_codomain(self, iteration):
        """
        Test that the nested tuple of modes refines the codomain.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        assert f.refined_codomain().refines(NestedTuple(f.codomain))

    @pytest.mark.parametrize("iteration", iterations)
    def test_from_refinement_roundtrip(self, iteration):
        """
        Test that from_refinement inverts refined_codomain: rebuilding a
        morphism from its modes-as-refinement recovers the morphism.

        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Fact_morphism()
        rebuilt = Fact_morphism.from_refinement(
            f.refined_codomain(), NestedTuple(f.codomain)
        )
        assert rebuilt == f

    def test_from_refinement_rejects_non_flat(self):
        """
        Test that from_refinement rejects a non-flat coarse tuple and a
        refinement with non-flat relative modes.
        """
        with pytest.raises(ValueError, match="flat"):
            Fact_morphism.from_refinement(
                NestedTuple(((2, 2), (3,))), NestedTuple(((4,), 3))
            )
        with pytest.raises(ValueError, match="flat"):
            Fact_morphism.from_refinement(
                NestedTuple((((2, 2), 2), (3,))), NestedTuple((8, 3))
            )
