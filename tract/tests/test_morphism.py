"""
Backend-free tests of the categorical properties of Fin, Tuple, and Nest
morphisms. Cross-validation against CuTe layouts lives in
test_cross_validation.py.
"""

import numpy as np
import pytest

from tract import FinMorphism, TupleMorphism, NestedTuple

from .conftest import seed_rngs
from .generators import (
    random_Tuple_morphism,
    random_complementable_Tuple_morphism,
    random_complementable_Nest_morphism,
)

iterations = range(100)
RANDOM_SEED_BASE = 42


class TestTupleMorphism:
    """Categorical properties of TupleMorphism operations."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_sort_is_sorted(self, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism()
        assert f.sort().is_sorted()

    @pytest.mark.parametrize("iteration", iterations)
    def test_coalesce_is_coalesced(self, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        assert f.coalesce().is_coalesced()

    @pytest.mark.parametrize("iteration", iterations)
    def test_complement_is_a_complement(self, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Tuple_morphism()
        assert f.is_complementary_to(f.complement())


class TestNestMorphism:
    """Categorical properties of NestMorphism operations."""

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_complement_is_a_complement(self, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Nest_morphism()
        assert f.is_complementary_to(f.complement())


class TestMorphismProperties:
    """Property-based tests for morphism operations."""

    def test_composition_associativity(self):
        seed_rngs(RANDOM_SEED_BASE)

        f = random_Tuple_morphism(max_length=5, max_value=10)
        g = random_Tuple_morphism(domain=f.codomain, max_length=5, max_value=10)
        h = random_Tuple_morphism(domain=g.codomain, max_length=5, max_value=10)

        # (h ∘ g) ∘ f = h ∘ (g ∘ f)
        left = f.compose(g).compose(h)
        right = f.compose(g.compose(h))
        assert left == right

    def test_identity_morphism(self):
        seed_rngs(RANDOM_SEED_BASE)

        n = 5
        domain = tuple(np.random.randint(1, 10) for _ in range(n))
        identity = TupleMorphism.identity(domain)

        assert identity.is_identity()
        assert identity.is_isomorphism()

        f = random_Tuple_morphism(codomain=domain, max_length=8, max_value=10)
        assert f.compose(identity) == f

    def test_complement_involution(self):
        """Complement is an involution, up to sorting."""
        seed_rngs(RANDOM_SEED_BASE)

        f = random_complementable_Tuple_morphism(max_length=6, max_value=10)
        f_comp_comp = f.complement().complement()
        sorted_f = f.sort()

        assert sorted_f.domain == f_comp_comp.domain
        assert sorted_f.codomain == f_comp_comp.codomain
        assert set(sorted_f.map) == set(f_comp_comp.map)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_morphism(self):
        f = TupleMorphism(tuple(), (3, 4), tuple())
        assert f.size() == 1
        assert f.cosize() == 12

        g = FinMorphism(3, 0, (0, 0, 0))
        assert g.domain == 3
        assert g.codomain == 0

    def test_single_element_morphism(self):
        f = TupleMorphism((5,), (5,), (1,))
        assert f.is_isomorphism()
        assert f.is_sorted()
        assert f.is_coalesced()

    def test_large_morphism(self):
        max_val = 2**30 - 1
        f = TupleMorphism(
            domain=(max_val, 2),
            codomain=(2, max_val),
            map=(2, 1),
        )
        assert f.size() == max_val * 2
        assert f.cosize() == max_val * 2

    def test_nested_tuple_edge_cases(self):
        nt1 = NestedTuple(5)
        assert nt1.rank() == 1
        assert nt1.length() == 1
        assert nt1.size() == 5

        nt2 = NestedTuple(((((2,),),),))
        assert nt2.length() == 1
        assert nt2.flatten() == (2,)

        nt3 = NestedTuple(())
        assert nt3.length() == 0
        assert nt3.size() == 1


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_sort_then_coalesce(self):
        seed_rngs(RANDOM_SEED_BASE)

        f = random_Tuple_morphism(max_length=8, max_value=10)
        sorted_f = f.sort()
        coalesced_f = sorted_f.coalesce()

        assert sorted_f.is_sorted()
        assert coalesced_f.is_coalesced()
        assert f.size() == sorted_f.size() == coalesced_f.size()
        assert f.cosize() == sorted_f.cosize() == coalesced_f.cosize()

    def test_complex_composition_chain(self):
        seed_rngs(RANDOM_SEED_BASE)

        morphisms = []
        domain = tuple(np.random.randint(1, 5) for _ in range(3))

        for _ in range(4):
            f = random_Tuple_morphism(domain=domain, max_length=5, max_value=10)
            morphisms.append(f)
            domain = f.codomain

        result = morphisms[0]
        for m in morphisms[1:]:
            result = result.compose(m)

        assert result.domain == morphisms[0].domain
        assert result.codomain == morphisms[-1].codomain
