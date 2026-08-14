"""
Unit tests for API surface not covered by the property and cross-validation
suites: restriction/factorization, squeezing, strong coalescence,
codomain updates, wedge sums, and the small NestedTuple/NestMorphism
helpers.
"""

import pytest

from tract import (
    FinMorphism,
    NestedTuple,
    NestMorphism,
    TupleMorphism,
)

from .conftest import seed_rngs
from .generators import random_Tuple_morphism

iterations = range(50)
RANDOM_SEED_BASE = 42


class TestFinMorphism:
    def test_identity(self):
        f = FinMorphism.identity(4)
        assert f.is_identity()
        assert f.map == (1, 2, 3, 4)

    def test_equality_ignores_name(self):
        f = FinMorphism(2, 3, (1, 3), name="f")
        g = FinMorphism(2, 3, (1, 3), name="g")
        assert f == g
        assert hash(f) == hash(g)
        assert f != FinMorphism(2, 3, (3, 1))

    def test_wedge(self):
        alpha = FinMorphism(2, 4, (1, 3))
        beta = FinMorphism(1, 4, (2,))
        wedge = alpha.wedge(beta)
        assert wedge == FinMorphism(3, 4, (1, 3, 2))

    def test_wedge_requires_same_codomain(self):
        with pytest.raises(ValueError):
            FinMorphism(1, 3, (1,)).wedge(FinMorphism(1, 4, (2,)))

    def test_wedge_requires_disjoint_images(self):
        with pytest.raises(ValueError):
            FinMorphism(1, 3, (1,)).wedge(FinMorphism(1, 3, (1,)))


class TestRestrictFactorizeSqueeze:
    def test_restrict(self):
        f = TupleMorphism((4, 6, 8), (8, 6, 4), (3, 2, 1))
        r = f.restrict((1, 3))
        assert r == TupleMorphism((4, 8), (8, 6, 4), (3, 1))

    def test_restrict_rejects_bad_indices(self):
        f = TupleMorphism((4, 6), (4, 6), (1, 2))
        with pytest.raises(ValueError):
            f.restrict((0, 1))
        with pytest.raises(ValueError):
            f.restrict((2, 1))

    def test_factorize(self):
        f = TupleMorphism((4, 8), (4, 6, 8), (1, 3))
        r = f.factorize((1, 3))
        assert r == TupleMorphism((4, 8), (4, 8), (1, 2))

    def test_squeeze_removes_unit_modes(self):
        f = TupleMorphism((1, 4, 1), (4, 1, 5), (0, 1, 0))
        s = f.squeeze()
        assert s == TupleMorphism((4,), (4, 5), (1,))

    @pytest.mark.parametrize("iteration", iterations)
    def test_squeeze_random(self, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        s = f.squeeze()
        assert 1 not in s.domain
        assert 1 not in s.codomain
        assert s.size() == f.size()


class TestStrongCoalesce:
    @pytest.mark.parametrize("iteration", iterations)
    def test_strong_coalesce_is_coalesced(self, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        c = f.strong_coalesce()
        assert c.is_coalesced()
        assert c.size() == f.size()

    def test_strong_coalesce_merges_across_unhit_codomain(self):
        # Weak coalescence keeps codomain modes separated by unhit modes;
        # strong coalescence merges through them.
        f = TupleMorphism((2, 3), (2, 5, 3), (1, 3))
        weak = f.coalesce()
        strong = f.strong_coalesce()
        assert len(strong.codomain) <= len(weak.codomain)


class TestCoalesceWithEquiv:
    @pytest.mark.parametrize("iteration", iterations)
    def test_matches_coalesce(self, iteration):
        seed_rngs(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        m, classes = f.coalesce_with_equiv()
        assert m == f.coalesce()
        # The classes partition the squeezed codomain in order.
        flat = [j for class_ in classes for j in class_]
        assert flat == sorted(flat)


class TestUpdateCodomain:
    def test_merges_codomain_modes(self):
        f = TupleMorphism((2, 4), (2, 3, 5, 4), (1, 4))
        g = f.update_codomain([[1], [2, 3], [4]])
        assert g == TupleMorphism((2, 4), (2, 15, 4), (1, 3))


class TestTupleMorphismIdentityEquality:
    def test_identity(self):
        f = TupleMorphism.identity((2, 3, 4))
        assert f.is_identity()
        assert f.compose(f) == f

    def test_equality_ignores_name(self):
        f = TupleMorphism((2,), (2, 3), (1,), name="a")
        g = TupleMorphism((2,), (2, 3), (1,), name="b")
        assert f == g
        assert hash(f) == hash(g)


class TestNestedTupleHelpers:
    def test_replace_empty_tuples(self):
        t = NestedTuple((2, (), (3, ())))
        assert t.replace_empty_tuples_with_one() == NestedTuple((2, 1, (3, 1)))
        assert t.replace_empty_tuples_with_zero() == NestedTuple((2, 0, (3, 0)))

    def test_equality_and_hash(self):
        assert NestedTuple((2, (3, 4))) == NestedTuple((2, (3, 4)))
        assert NestedTuple((2, (3, 4))) != NestedTuple((2, 3, 4))
        assert hash(NestedTuple(5)) == hash(NestedTuple(5))

    def test_sublength(self):
        T = NestedTuple((6, 6))
        refined = NestedTuple(((2, 3), (2, 3)))
        assert refined.refines(T)
        assert refined.sublength(1, T) == 0
        assert refined.sublength(2, T) == 2


class TestNestMorphismHelpers:
    def test_identity(self):
        T = NestedTuple((2, (3, 4)))
        f = NestMorphism.identity(T)
        assert f.is_identity()
        assert f.compose(f) == f

    def test_flatten_codomain(self):
        f = NestMorphism(
            NestedTuple((2, 3)), NestedTuple((2, (3, 5))), (1, 2)
        )
        g = f.flatten_codomain()
        assert g.domain == f.domain
        assert g.codomain == NestedTuple((2, 3, 5))
        assert g.map == f.map
        assert g.flatten() == f.flatten()

    def test_repr_in_tex(self):
        f = NestMorphism(NestedTuple((2,)), NestedTuple((2, 3)), (1,))
        tex = f.repr_in_tex()
        assert tex.startswith("$") and tex.endswith("$")
        assert "\\xrightarrow" in tex
