"""
Comprehensive test suite for code accompanying "Categorical Foundations for CuTe layouts" by Colfax Research.

Tests the mathematical properties and implementations of morphisms
in categories E_0, Tuple, and Nest, along with their associated
layout computations.
"""

import numpy as np
import pytest
import cutlass
import cutlass.cute as cute

# Import from the tract package
from tract.categories import (
    Fin_morphism,
    Tuple_morphism,
    Nest_morphism,
    NestedTuple
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

from tract.layout_utils import (
    compute_flat_layout,
    compute_layout,
    flatten_layout,
    flat_concatenate,
    concatenate,
    nullify_trivial_strides,
    nullify_zero_strides,
    mutual_refinement,
)

# Test configuration
iterations = range(100)
RANDOM_SEED_BASE = 42


# *************************************************************************
# TUPLE_MORPHISM TEST COMPONENTS
# *************************************************************************

@cute.jit
def coalesce_agree(f: cutlass.Constexpr[Tuple_morphism]) -> bool:
    """
    Check if morphism coalescence agrees with layout coalescence.
    
    :param f: Tuple morphism to test
    :type f: Tuple_morphism
    :return: True if coalescence operations agree
    :rtype: bool
    """
    coalesce_f = f.coalesce()
    layout_f = compute_flat_layout(f)
    coalesce_layout = compute_flat_layout(coalesce_f)
    layout_coalesce = cute.coalesce(layout_f)
    
    agree = (
        (coalesce_layout == layout_coalesce)
        or (
            cute.rank(coalesce_layout) == 0
            and layout_coalesce == cute.make_layout(1, stride=0)
        )
        or (
            cute.rank(coalesce_layout) == 1
            and layout_coalesce
            == cute.make_layout(
                coalesce_layout.shape[0], stride=coalesce_layout.stride[0]
            )
        )
    )
    return agree

@cute.jit
def concat_agree(
    f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]
) -> bool:
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

@cute.jit
def compose_agree(
    f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]
) -> bool:
    """
    Check if morphism composition agrees with layout composition.
    
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
    compose_layout = cute.composition(layout_g, layout_f)
    return layout_compose == compose_layout

@cute.jit
def complement_agree(f: cutlass.Constexpr[Tuple_morphism]) -> bool:
    """
    Check if morphism complement agrees with layout complement.
    
    :param f: Morphism to test
    :type f: Tuple_morphism
    :return: True if complements agree
    :rtype: bool
    """
    f_complement = f.complement()
    layout_f = compute_flat_layout(f)
    layout_f_complement = compute_flat_layout(f_complement)
    complement_layout_f = cute.complement(layout_f, f.cosize())
    return cute.coalesce(complement_layout_f) == cute.coalesce(layout_f_complement)

@cute.jit
def flat_divide_agree(
    f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]
) -> bool:
    """
    Check if flat division of morphisms agrees with layout division.
    
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
    quotient_layout = flatten_layout(cute.logical_divide(layout_f, layout_g))
    layout_quotient = compute_flat_layout(quotient)
    return cute.coalesce(layout_quotient) == cute.coalesce(quotient_layout)

@cute.jit
def flat_product_agree(
    f: cutlass.Constexpr[Tuple_morphism], g: cutlass.Constexpr[Tuple_morphism]
) -> bool:
    """
    Check if flat product of morphisms agrees with layout product.
    
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
    product = flatten_layout(cute.logical_product(A, B))
    return C == product


# *************************************************************************
# TUPLE_MORPHISM TESTS
# *************************************************************************


class TestTupleMorphism:
    """Test suite for Tuple_morphism operations."""
    
    @pytest.mark.parametrize("iteration", iterations)
    def test_sort_is_sorted(self, iteration):
        """
        Test that sorting a tuple morphism produces a sorted morphism.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism()
        assert f.sort().is_sorted()

    @pytest.mark.parametrize("iteration", iterations)
    def test_coalesce_is_coalesced(self, iteration):
        """
        Test that coalescing a tuple morphism produces a coalesced morphism.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        assert f.coalesce().is_coalesced()

    @pytest.mark.parametrize("iteration", iterations)
    def test_complement_is_a_complement(self, iteration):
        """
        Test that complement of a complementable morphism is valid.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Tuple_morphism()
        assert f.is_complementary_to(f.complement())

    @pytest.mark.parametrize("iteration", iterations)
    def test_coalesce_agree(self, iteration):
        """
        Test that morphism coalescence agrees with layout coalescence.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Tuple_morphism(max_value=10)
        assert coalesce_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_concat_agree(self, iteration):
        """
        Test that morphism concatenation agrees with layout concatenation.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_Tuple_morphisms_with_disjoint_images()
            assert concat_agree(f, g)
        except OverflowError as e:
            pytest.skip(f"Skipped due to cosize overflow: {e}")

    @pytest.mark.parametrize("iteration", iterations)
    def test_compose_agree(self, iteration):
        """
        Test that morphism composition agrees with layout composition.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Tuple_morphisms()
        assert compose_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_complement_agree(self, iteration):
        """
        Test that morphism complement agrees with layout complement.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Tuple_morphism(max_value=10)
        assert complement_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_flat_divide_agree(self, iteration):
        """
        Test that flat division agrees between morphisms and layouts.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_divisible_Tuple_morphisms()
        assert flat_divide_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_flat_product_agree(self, iteration):
        """
        Test that flat product agrees between morphisms and layouts.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_product_admissible_Tuple_morphisms()
        assert flat_product_agree(f, g)


# *************************************************************************
# NEST_MORPHISM TEST COMPONENTS
# *************************************************************************

@cute.jit
def Nest_concat_agree(
    f: cutlass.Constexpr[Nest_morphism], g: cutlass.Constexpr[Nest_morphism]
) -> bool:
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

@cute.jit
def Nest_complement_agree(f: cutlass.Constexpr[Nest_morphism]) -> bool:
    """
    Check if nested tuple morphism complement agrees with layout complement.
    
    :param f: Morphism to test
    :type f: Nest_morphism
    :return: True if complements agree
    :rtype: bool
    """
    f_complement = f.complement()
    layout_f = compute_layout(f)
    layout_f_complement = compute_layout(f_complement)
    complement_layout_f = cute.complement(layout_f, f.cosize())
    return complement_layout_f == cute.coalesce(layout_f_complement)

@cute.jit
def Nest_compose_agree(
    f: cutlass.Constexpr[Nest_morphism], g: cutlass.Constexpr[Nest_morphism]
) -> bool:
    """
    Check if nested morphism composition agrees with layout composition.
    
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
    compose_layout = cute.composition(layout_g, layout_f)
    return layout_compose == compose_layout

@cute.jit
def Nest_coalesce_agree(f: cutlass.Constexpr[Nest_morphism]) -> bool:
    """
    Check if nested morphism coalescence agrees with layout coalescence.
    
    :param f: Nest morphism to test
    :type f: Nest_morphism
    :return: True if coalesce operations agree
    :rtype: bool
    """
    coalesce_f = f.coalesce()
    layout_f = compute_layout(f)
    coalesce_layout = compute_layout(coalesce_f)
    layout_coalesce = cute.coalesce(layout_f)
    agree = (
        (coalesce_layout == layout_coalesce)
        or (
            cute.rank(layout_coalesce) == 0
            and coalesce_layout == cute.make_layout(1, stride=0)
        )
    )
    return agree

@cute.jit
def Nest_logical_product_agree(f: cutlass.Constexpr[Nest_morphism], g: cutlass.Constexpr[Nest_morphism]):
    layout_f = compute_layout(f)
    layout_g = compute_layout(g)
    product = f.logical_product(g)
    product_layout = cute.logical_product(layout_f, layout_g)
    layout_product = compute_layout(product)
    return layout_product == product_layout

@cute.jit
def Nest_logical_divide_agree(f: cutlass.Constexpr[Nest_morphism], g: cutlass.Constexpr[Nest_morphism]):
    morphism_quotient = f.logical_divide(g)
    layout_quotient = compute_layout(morphism_quotient)
    layout_f = compute_layout(f)
    layout_g = compute_layout(g)
    layout_g_complement = cute.complement(layout_g,cute.size(layout_f))
    quotient_layout = cute.composition(layout_f,concatenate(layout_g,layout_g_complement))
    return cute.coalesce(layout_quotient)==cute.coalesce(quotient_layout)

@cute.jit
def composition_algorithm_agree(
    f: cutlass.Constexpr[Nest_morphism], g: cutlass.Constexpr[Nest_morphism]
) -> bool:
    """
    Check if weak composition algorithm produces correct result.
    
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
    V = g.codomain
    
    Tprime, Uprime = mutual_refinement(T, U)
    fprime = f.pullback_along(Tprime)
    inclusion = Nest_morphism(Tprime, Uprime, tuple(range(1, Tprime.length() + 1)))
    gprime = g.pushforward_along(Uprime)
    weak_composite = compute_layout(fprime.compose(inclusion).compose(gprime))
    composite = cute.coalesce(weak_composite, target_profile=S.data)
    return composite == cute.composition(compute_layout(g), compute_layout(f))


# *************************************************************************
# NEST_MORPHISM TESTS
# *************************************************************************


class TestNestMorphism:
    """Test suite for Nest_morphism operations."""
    
    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_complement_is_a_complement(self, iteration):
        """
        Test that complement of a complementable nested morphism is valid.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Nest_morphism()
        assert f.is_complementary_to(f.complement())

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_complement_agree(self, iteration):
        """
        Test that nested tuple morphism complement agrees with layout complement.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_complementable_Nest_morphism(max_value=10)
        assert Nest_complement_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_concat_agree(self, iteration):
        """
        Test that nested morphism concatenation agrees with layout.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        try:
            f, g = random_Nest_morphisms_with_disjoint_images()
            assert Nest_concat_agree(f, g)
        except OverflowError as e:
            pytest.skip(f"Skipped due to cosize overflow: {e}")

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_coalesce_agree(self, iteration):
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f = random_Nest_morphism(max_value=10)
        assert Nest_coalesce_agree(f)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_logical_divide_agree(self,iteration):
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f,g = random_divisible_Nest_morphisms()
        assert Nest_logical_divide_agree(f,g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_logical_product_agree(self,iteration):
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f,g = random_product_admissible_Nest_morphisms()
        assert Nest_logical_product_agree(f,g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_Nest_compose_agree(self, iteration):
        """
        Test that nested morphism composition agrees with layout.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        f, g = random_composable_Nest_morphisms(
            min_length=0, max_length=6, max_value=64
        )
        assert Nest_compose_agree(f, g)

    @pytest.mark.parametrize("iteration", iterations)
    def test_composition_algorithm_agree(self, iteration):
        """
        Test that weak composition algorithm is correct.
        
        :param iteration: Test iteration number for seeding
        :type iteration: int
        """
        np.random.seed(RANDOM_SEED_BASE + iteration)
        T, U = random_mutually_refinable_nested_tuples()
        f = random_Nest_morphism(codomain=T, max_length=8, max_value=16)
        g = random_Nest_morphism(domain=U, max_length=8, max_value=16)
        assert composition_algorithm_agree(f, g)


# *************************************************************************
# PROPERTY-BASED TESTS
# *************************************************************************


class TestMorphismProperties:
    """Property-based tests for morphism operations."""
    
    def test_composition_associativity(self):
        """Test that morphism composition is associative."""
        np.random.seed(RANDOM_SEED_BASE)
        
        # Create three composable morphisms
        f = random_Tuple_morphism(max_length=5, max_value=10)
        g = random_Tuple_morphism(domain=f.codomain, max_length=5, max_value=10)
        h = random_Tuple_morphism(domain=g.codomain, max_length=5, max_value=10)
        
        # Test (h ∘ g) ∘ f = h ∘ (g ∘ f)
        left = f.compose(g).compose(h)
        right = f.compose(g.compose(h))
        
        assert left.domain == right.domain
        assert left.codomain == right.codomain
        assert left.map == right.map
    
    def test_identity_morphism(self):
        """Test that identity morphism behaves correctly."""
        np.random.seed(RANDOM_SEED_BASE)
        
        # Create identity morphism
        n = 5
        domain = tuple(np.random.randint(1, 10) for _ in range(n))
        identity = Tuple_morphism(domain, domain, tuple(range(1, n + 1)))
        
        # Test that it's an isomorphism
        assert identity.is_isomorphism()
        
        # Test that composing with identity gives same morphism
        f = random_Tuple_morphism(codomain=domain, max_length=8, max_value=10)
        assert f.compose(identity).map == f.map
        
    def test_complement_involution(self):
        """Test that complement is an involution, up to sorting."""
        np.random.seed(RANDOM_SEED_BASE)
        
        f = random_complementable_Tuple_morphism(max_length=6, max_value=10)
        f_comp = f.complement()
        f_comp_comp = f_comp.complement()
        sorted_f = f.sort()
        
        # Complement of complement should give back original
        assert sorted_f.domain == f_comp_comp.domain
        assert sorted_f.codomain == f_comp_comp.codomain
        assert set(sorted_f.map) == set(f_comp_comp.map)


# *************************************************************************
# EDGE CASE TESTS
# *************************************************************************


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_morphism(self):
        """Test morphisms with empty domain or codomain."""
        # Empty domain
        f = Tuple_morphism(tuple(), (3, 4), tuple())
        assert f.size() == 1
        assert f.cosize() == 12
        
        # Empty codomain
        g = Fin_morphism(3, 0, (0, 0, 0))
        assert g.domain == 3
        assert g.codomain == 0
    
    def test_single_element_morphism(self):
        """Test morphisms with single elements."""
        f = Tuple_morphism((5,), (5,), (1,))
        assert f.is_isomorphism()
        assert f.is_sorted()
        assert f.is_coalesced()
    
    def test_large_morphism(self):
        """Test morphisms with large values."""
        np.random.seed(RANDOM_SEED_BASE)
        
        # Test with values near overflow limit
        max_val = 2**30 - 1
        f = Tuple_morphism(
            domain=(max_val, 2),
            codomain=(2, max_val),
            map=(2, 1)
        )
        
        # Should not overflow
        assert f.size() == max_val * 2
        assert f.cosize() == max_val * 2
    
    def test_nested_tuple_edge_cases(self):
        """Test edge cases for nested tuples."""
        # Single integer
        nt1 = NestedTuple(5)
        assert nt1.rank() == 1
        assert nt1.length() == 1
        assert nt1.size() == 5
        
        # Deeply nested
        nt2 = NestedTuple(((((2,),),),))
        assert nt2.length() == 1
        assert nt2.flatten() == (2,)
        
        # Empty tuple
        nt3 = NestedTuple(())
        assert nt3.length() == 0
        assert nt3.size() == 1


# *************************************************************************
# INTEGRATION TESTS
# *************************************************************************


class TestIntegration:
    """Integration tests combining multiple operations."""
    
    def test_sort_then_coalesce(self):
        """Test sorting followed by coalescing."""
        np.random.seed(RANDOM_SEED_BASE)
        
        f = random_Tuple_morphism(max_length=8, max_value=10)
        sorted_f = f.sort()
        coalesced_f = sorted_f.coalesce()
        
        assert sorted_f.is_sorted()
        assert coalesced_f.is_coalesced()
        
        # Both should have same size/cosize
        assert f.size() == sorted_f.size() == coalesced_f.size()
        assert f.cosize() == sorted_f.cosize() == coalesced_f.cosize()
    
    def test_complex_composition_chain(self):
        """Test a complex chain of compositions."""
        np.random.seed(RANDOM_SEED_BASE)
        
        # Create a chain of morphisms
        morphisms = []
        domain = tuple(np.random.randint(1, 5) for _ in range(3))
        
        for _ in range(4):
            f = random_Tuple_morphism(
                domain=domain, max_length=5, max_value=10
            )
            morphisms.append(f)
            domain = f.codomain
        
        # Compose them all
        result = morphisms[0]
        for m in morphisms[1:]:
            result = result.compose(m)
        
        assert result.domain == morphisms[0].domain
        assert result.codomain == morphisms[-1].codomain


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])