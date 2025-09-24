"""
Test utilities for Tract library.

This module provides random generation functions for morphisms and nested tuples,
used for property-based testing and validation.
"""

import random
from math import prod
import numpy as np

try:
    import cutlass
    import cutlass.cute as cute
except ImportError:
    cutlass = None
    cute = None

from .categories import (
    Fin_morphism,
    Tuple_morphism,
    Nest_morphism,
    NestedTuple
)


# *************************************************************************
# RANDOM GENERATION FUNCTIONS
# *************************************************************************


def random_ordered_subtuple(m: int, k: int) -> tuple:
    """
    Generate a tuple of k distinct integers sampled uniformly from {1,...,m}.
    
    :param m: Upper bound of the set (inclusive)
    :type m: int
    :param k: Number of distinct elements to sample
    :type k: int
    :return: Tuple of k distinct integers from {1,...,m}
    :rtype: tuple
    :raises ValueError: If k not in range [0, m]
    """
    if not (0 <= k <= m):
        raise ValueError("k must satisfy 0 <= k <= m")

    sampled = random.sample(range(1, m + 1), k)
    return tuple(sampled)


def random_Fin_morphism(domain=None, codomain=None, min_length=0, max_length=9):
    """
    Generate a random morphism in Fin_*.
    
    :param domain: Domain size (optional)
    :type domain: int or None
    :param codomain: Codomain size (optional)
    :type codomain: int or None
    :param min_length: Minimum length for random generation
    :type min_length: int
    :param max_length: Maximum length for random generation
    :type max_length: int
    :return: Random Fin morphism
    :rtype: Fin_morphism
    """
    if domain is None:
        domain = np.random.randint(min_length, max_length + 1)
    if codomain is None:
        codomain = np.random.randint(min_length, max_length + 1)

    permutation1_map = tuple(
        int(x) for x in np.random.permutation(range(1, domain + 1))
    )
    permutation1 = Fin_morphism(domain, domain, permutation1_map)

    max_size = min(domain, codomain)
    u = np.random.rand()
    skewed = 1 - u**3  # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    
    projection_map = list(range(1, size_of_image + 1))
    for _ in range(domain - size_of_image):
        projection_map.append(0)
    projection_map = tuple(projection_map)
    projection = Fin_morphism(domain, size_of_image, projection_map)

    inclusion_map = tuple(range(1, size_of_image + 1))
    inclusion = Fin_morphism(size_of_image, codomain, inclusion_map)

    permutation2_map = tuple(
        int(x) for x in np.random.permutation(range(1, codomain + 1))
    )
    permutation2 = Fin_morphism(codomain, codomain, permutation2_map)

    return permutation1.compose(projection).compose(inclusion).compose(permutation2)


def random_complementable_Fin_morphism(
    domain=None, codomain=None, min_length=0, max_length=9
):
    """
    Generate a random complementable morphism in Fin_*.
    
    :param domain: Domain size (optional)
    :type domain: int or None
    :param codomain: Codomain size (optional)
    :type codomain: int or None
    :param min_length: Minimum length for random generation
    :type min_length: int
    :param max_length: Maximum length for random generation
    :type max_length: int
    :return: Random complementable Fin morphism
    :rtype: Fin_morphism
    """
    if domain is None:
        domain = np.random.randint(min_length, max_length + 1)
    if codomain is None:
        codomain = np.random.randint(domain, max_length + 1)

    permutation1_map = tuple(
        int(x) for x in np.random.permutation(range(1, domain + 1))
    )
    permutation1 = Fin_morphism(domain, domain, permutation1_map)

    inclusion_map = tuple(range(1, domain + 1))
    inclusion = Fin_morphism(domain, codomain, inclusion_map)

    permutation2_map = tuple(
        int(x) for x in np.random.permutation(range(1, codomain + 1))
    )
    permutation2 = Fin_morphism(codomain, codomain, permutation2_map)

    return permutation1.compose(inclusion).compose(permutation2)


def random_Tuple_morphism(
    domain=None, codomain=None, min_length=0, max_length=9, min_value = 1, max_value=2**30-1, nondegenerate = True
):
    """
    Generate a random tuple morphism.
    
    :param domain: Domain tuple (optional)
    :type domain: tuple or None
    :param codomain: Codomain tuple (optional)  
    :type codomain: tuple or None
    :param min_length: Minimum length for random generation
    :type min_length: int
    :param max_length: Maximum length for random generation
    :type max_length: int
    :param min_value: Minimum value for tuple entries
    :type min_value: int
    :param max_value: Maximum value for tuple entries
    :type max_value: int
    :return: Random tuple morphism
    :rtype: Tuple_morphism
    """
    MAX_VALUE = 2**31 - 1
    assert (domain is None) or (codomain is None)

    if domain is not None:
        domain_length = len(domain)
        codomain_length = np.random.randint(min_length, max_length)
        underlying_map = random_Fin_morphism(domain_length, codomain_length).map
        if nondegenerate:
            underlying_map = list(underlying_map)
            for i in range(1, domain_length+1):
                if domain[i-1] == 1:
                    underlying_map[i-1] = 0
            underlying_map = tuple(underlying_map)
        codomain = []
        cosize = 1
        
        for j in range(1, codomain_length + 1):
            if j not in underlying_map:
                codomain.append(np.random.randint(min_value, min(max_value, (2**30 - 1) // cosize)))
            else:
                codomain.append(domain[underlying_map.index(j)])
            cosize *= codomain[-1]
        codomain = tuple(codomain)
        return Tuple_morphism(domain, codomain, underlying_map)

    else:
        domain_length = int(np.random.randint(min_length, max_length))

        if codomain is None:
            codomain_length = np.random.randint(min_length, max_length)
            codomain = []
            cosize = 1
            for _ in range(codomain_length):
                upper = min(max_value, max(min_value, MAX_VALUE // cosize))
                codomain.append(np.random.randint(min_value, min(max_value, upper) + 1))
                cosize *= codomain[-1]
            codomain = tuple(codomain)
            
        codomain_length = len(codomain)

        underlying_Fin_morphism = random_Fin_morphism(
            domain=domain_length, codomain=codomain_length
        )
        underlying_map = underlying_Fin_morphism.map
        domain = []
        
        for i in range(1, domain_length + 1):
            if underlying_map[i - 1] > 0:
                domain.append(codomain[underlying_map[i - 1] - 1])
            else:
                domain.append(np.random.randint(min_value, max_value))
        domain = tuple(domain)

        if nondegenerate:
            underlying_map = list(underlying_map)
            for i in range(1, domain_length+1):
                if domain[i-1] == 1:
                    underlying_map[i-1] = 0
            underlying_map = tuple(underlying_map)
        
        assert prod(codomain) < MAX_VALUE
        return Tuple_morphism(domain, codomain, underlying_map)


def random_composable_Tuple_morphisms(min_length=0, max_length=9, min_value = 1, max_value=10, nondegenerate = True):
    """
    Generate a pair of composable tuple morphisms.
    
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Pair of composable morphisms (f, g)
    :rtype: Tuple[Tuple_morphism, Tuple_morphism]
    """
    f = random_Tuple_morphism(
        min_length=min_length, max_length=max_length, min_value = min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    g = random_Tuple_morphism(
        domain=f.codomain,
        min_length=min_length,
        max_length=max_length,
        min_value=min_value,
        max_value=max_value,
        nondegenerate=nondegenerate
    )
    return f, g


def random_Tuple_morphisms_with_disjoint_images(
    min_length=0, max_length=9, min_value = 1, max_value=10, nondegenerate = True
):
    """
    Generate a pair of tuple morphisms with same codomain and disjoint images.
    
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Pair of morphisms with disjoint images
    :rtype: Tuple[Tuple_morphism, Tuple_morphism]
    """
    f = random_Tuple_morphism(min_length=min_length, max_length=max_length, min_value = min_value, max_value=max_value, nondegenerate=nondegenerate)
    codomain = f.codomain
    codomain_length = len(codomain)

    domain_length = np.random.randint(min_length, max_length + 1)
    domain = []
    for i in range(domain_length):
        domain.append(np.random.randint(min_value, max_value + 1))

    # Compute possible values for the map beta underlying g
    possible_values = []
    for j in range(1, codomain_length + 1):
        if j not in f.map:
            possible_values.append(j)

    # Select a random subset of those possible values
    max_size = min(domain_length, len(possible_values))
    u = np.random.rand()
    skewed = 1 - u**3  # skew toward larger values
    size_of_image = int(np.round(skewed * max_size))
    size_of_image = max(0, min(size_of_image, max_size))

    image = [
        int(x) for x in np.random.choice(possible_values, size_of_image, replace=False)
    ]
    map_ = [0] * domain_length
    surviving_indices = sorted(random_ordered_subtuple(domain_length, size_of_image))
    
    for i, index in enumerate(surviving_indices):
        map_[index - 1] = image[i]
    map_ = tuple(map_)

    for i in range(len(domain)):
        if map_[i] > 0:
            domain[i] = codomain[map_[i] - 1]
    domain = tuple(domain)

    if nondegenerate:
        map_ = list(map_)
        for i in range(1, domain_length+1):
            if domain[i-1] == 1:
                map_[i-1] = 0
        map_ = tuple(map_)

    g = Tuple_morphism(domain, codomain, map_)
    return f, g


def random_complementable_Tuple_morphism(min_length=2, max_length=9, min_value=2,  max_value=10, nondegenerate = True):
    """
    Generate a random complementable tuple morphism.
    
    Creates a morphism with 0 < length(domain) < length(codomain).
    
    :param min_length: Minimum codomain length
    :type min_length: int
    :param max_length: Maximum codomain length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Random complementable morphism
    :rtype: Tuple_morphism
    """
    if nondegenerate:
        min_value = max(2, min_value)

    codomain_length = np.random.randint(min_length, max_length + 1)
    codomain = []
    cosize = 1
    
    for _ in range(codomain_length):
        new_int = np.random.randint(min_value, max_value + 1)
        cosize *= new_int
        if cosize < 2**30 - 1:  # Check for potential overflow
            codomain.append(new_int)
    codomain = tuple(codomain)

    size_of_image = np.random.randint(1, len(codomain))
    map_ = random_ordered_subtuple(len(codomain), size_of_image)

    domain = []
    for value in map_:
        domain.append(codomain[value - 1])
    domain = tuple(domain)

    return Tuple_morphism(domain, codomain, map_)


def random_divisible_Tuple_morphisms(min_length=2, max_length=9, min_value=1, max_value=10, nondegenerate = True):
    """
    Generate a pair of tuple morphisms f, g where g divides f.
    
    g is injective and codomain(g) = domain(f).
    
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Pair (f, g) where g divides f
    :rtype: Tuple[Tuple_morphism, Tuple_morphism]
    """
    if nondegenerate:
        min_value = max(2, min_value)

    f = random_Tuple_morphism(
        min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    codomain = f.domain

    domain_length = np.random.randint(1, len(codomain) + 1)
    domain = [0] * domain_length

    # Generate map
    map_ = random_ordered_subtuple(len(codomain), len(domain))

    for i, value in enumerate(map_):
        if value > 0:
            domain[i] = codomain[value - 1]
    domain = tuple(domain)

    g = Tuple_morphism(domain, codomain, map_)
    return f, g


def random_product_admissible_Tuple_morphisms(min_length=2, max_length=9, min_value=1, max_value=10, nondegenerate = True):
    """
    Generate a pair of tuple morphisms admissible for product operation.
    
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Pair of product-admissible morphisms
    :rtype: Tuple[Tuple_morphism, Tuple_morphism]
    """
    if nondegenerate:
        min_value = max(2, min_value)

    f = random_complementable_Tuple_morphism(
        min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    codomain = f.complement().domain

    domain = []
    size_of_image = np.random.randint(0, len(codomain) + 1)
    m = size_of_image + np.random.randint(0, 5)
    
    for _ in range(m):
        domain.append(np.random.randint(min_value, max_value + 1))
        
    image = random_ordered_subtuple(len(codomain), size_of_image)
    map_ = [0 for _ in range(len(domain))]
    surviving_indices = sorted(random_ordered_subtuple(len(domain), size_of_image))
    
    for i, index in enumerate(surviving_indices):
        map_[index - 1] = image[i]

    for i, value in enumerate(map_):
        if value > 0:
            domain[i] = codomain[value - 1]
    domain = tuple(domain)
    map_ = tuple(map_)

    if nondegenerate:
        map_ = list(map_)
        for i in range(1, len(domain)+1):
            if domain[i-1] == 1:
                map_[i-1] = 0
        map_ = tuple(map_)

    g = Tuple_morphism(domain, codomain, map_)
    return f, g


def random_NestedTuple(length=5, max_depth=4, max_width=5, min_value=1, max_value=10):
    """
    Generate a random NestedTuple containing exactly `length` integers.
    
    :param length: Number of integers in flattened tuple
    :type length: int
    :param max_depth: Maximum nesting depth
    :type max_depth: int
    :param max_width: Maximum width at each level
    :type max_width: int
    :param int_range: Range for integer values
    :type int_range: tuple
    :return: Random nested tuple
    :rtype: NestedTuple
    """
    int_range = (min_value,max_value)
    def generate(depth, remaining):
        if remaining <= 0:
            return ()

        if depth >= max_depth:
            return tuple(random.randint(*int_range) for _ in range(remaining))

        if remaining == 1:
            return random.randint(*int_range)

        valid_width = min(max_width, remaining)
        if valid_width < 1:
            return ()

        width = random.randint(1, valid_width)

        if width == 1:
            counts = [remaining]
        else:
            cuts = sorted(random.sample(range(1, remaining), width - 1))
            counts = (
                [cuts[0]]
                + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))]
                + [remaining - cuts[-1]]
            )

        return tuple(generate(depth + 1, count) for count in counts)

    return NestedTuple(generate(0, length))


def random_profile(length=5, max_depth=4, max_width=5):
    """
    Generate a random profile (nested tuple with all zeros).
    
    :param length: Number of zeros in flattened tuple
    :type length: int
    :param max_depth: Maximum nesting depth
    :type max_depth: int
    :param max_width: Maximum width at each level
    :type max_width: int
    :return: Random profile
    :rtype: NestedTuple
    """
    return random_NestedTuple(
        length=length, max_depth=max_depth, max_width=max_width, min_value=0, max_value=0
    )


def random_Nest_morphism(
    domain=None, codomain=None, min_length=0, max_length=9, min_value=1, max_value=128, nondegenerate = True
):
    """
    Generate a random nested tuple morphism.
    
    :param domain: Domain nested tuple (optional)
    :type domain: NestedTuple or None
    :param codomain: Codomain nested tuple (optional)
    :type codomain: NestedTuple or None
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Random nested tuple morphism
    :rtype: Nest_morphism
    """
    assert (domain is None) or (codomain is None)
    
    if domain is not None:
        flat_morphism = random_Tuple_morphism(
            domain=domain.flatten(),
            min_length=min_length,
            max_length=max_length,
            min_value=min_value,
            max_value=max_value,
            nondegenerate=nondegenerate
        )
        flat_codomain = flat_morphism.codomain
        codomain = random_profile(length=len(flat_codomain)).sub(flat_codomain)
    elif codomain is not None:
        flat_morphism = random_Tuple_morphism(
            codomain=codomain.flatten(),
            min_length=min_length,
            max_length=max_length,
            min_value=min_value,
            max_value=max_value,
            nondegenerate=nondegenerate
        )
        flat_domain = flat_morphism.domain
        domain = random_profile(length=len(flat_domain)).sub(flat_domain)
    else:
        flat_morphism = random_Tuple_morphism(
            min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
        )
        flat_domain = flat_morphism.domain
        flat_codomain = flat_morphism.codomain
        domain = random_profile(length=len(flat_domain)).sub(flat_domain)
        codomain = random_profile(length=len(flat_codomain)).sub(flat_codomain)
        
    return Nest_morphism(domain, codomain, flat_morphism.map)


def random_composable_Nest_morphisms(min_length=0, max_length=10, min_value=1, max_value=1024, nondegenerate = True):
    """
    Generate a pair of composable nested tuple morphisms.
    
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Pair of composable morphisms
    :rtype: Tuple[Nest_morphism, Nest_morphism]
    """
    f = random_Nest_morphism(
        min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    g = random_Nest_morphism(
        domain=f.codomain,
        min_length=min_length,
        max_length=max_length,
        min_value=min_value,
        max_value=max_value,
        nondegenerate=nondegenerate
    )
    return f, g


def random_Nest_morphisms_with_disjoint_images(min_length=0, max_length=9, min_value=1, max_value=10, nondegenerate = True):
    """
    Generate a pair of nested tuple morphisms with disjoint images.
    
    :return: Pair of morphisms with disjoint images
    :rtype: Tuple[Nest_morphism, Nest_morphism]
    """
    flat_f, flat_g = random_Tuple_morphisms_with_disjoint_images(
        min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    domain_f = random_profile(length=len(flat_f.domain)).sub(flat_f.domain)
    codomain_f = random_profile(length=len(flat_f.codomain)).sub(flat_f.codomain)
    domain_g = random_profile(length=len(flat_g.domain)).sub(flat_g.domain)
    codomain_g = codomain_f
    
    f = Nest_morphism(domain_f, codomain_f, flat_f.map)
    g = Nest_morphism(domain_g, codomain_g, flat_g.map)
    return f, g


def random_complementable_Nest_morphism(min_length=2, max_length=9, min_value=1, max_value=10, nondegenerate = True):
    """
    Generate a random complementable nested tuple morphism.
    
    :return: Random complementable morphism
    :rtype: Nest_morphism
    """
    flat_f = random_complementable_Tuple_morphism(
        min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    domain_f = random_NestedTuple(length=len(flat_f.domain)).sub(flat_f.domain)
    codomain_f = random_NestedTuple(length=len(flat_f.codomain)).sub(flat_f.codomain)
    map_f = flat_f.map
    return Nest_morphism(domain_f, codomain_f, map_f)


def random_divisible_Nest_morphisms(min_length=2, max_length=9, min_value=2, max_value=10, nondegenerate = True):
    """
    Generate a pair of nested tuple morphisms where one divides the other.
    
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param min_value: Minimum value for entries
    :type min_value: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Pair (f, g) where g divides f
    :rtype: Tuple[Nest_morphism, Nest_morphism]
    """
    flat_f, flat_g = random_divisible_Tuple_morphisms(
        min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    domain_f = random_profile(length=len(flat_f.domain)).sub(flat_f.domain)
    codomain_f = random_profile(length=len(flat_f.codomain)).sub(flat_f.codomain)
    map_f = flat_f.map
    
    domain_g = random_profile(length=len(flat_g.domain)).sub(flat_g.domain)
    codomain_g = NestedTuple(domain_f.data)
    map_g = flat_g.map
    
    f = Nest_morphism(domain_f, codomain_f, map_f)
    g = Nest_morphism(domain_g, codomain_g, map_g)
    return f, g


def random_product_admissible_Nest_morphisms(min_length=2, max_length=9, min_value=1, max_value=10, nondegenerate = True):
    """
    Generate a pair of nested tuple morphisms admissible for product.
    
    :param min_length: Minimum length
    :type min_length: int
    :param max_length: Maximum length
    :type max_length: int
    :param max_value: Maximum value for entries
    :type max_value: int
    :return: Pair of product-admissible morphisms
    :rtype: Tuple[Nest_morphism, Nest_morphism]
    """
    flat_f, flat_g = random_product_admissible_Tuple_morphisms(
        min_length=min_length, max_length=max_length, min_value=min_value, max_value=max_value, nondegenerate=nondegenerate
    )
    domain_f = random_profile(length=len(flat_f.domain)).sub(flat_f.domain)
    codomain_f = random_profile(length=len(flat_f.codomain)).sub(flat_f.codomain)
    map_f = flat_f.map
    
    domain_g = random_profile(length=len(flat_g.domain)).sub(flat_g.domain)
    # codomain_g = random_profile(length=len(flat_g.codomain)).sub(flat_g.codomain)
    codomain_g = NestedTuple(flat_g.codomain)
    map_g = flat_g.map
    
    f = Nest_morphism(domain_f, codomain_f, map_f)
    g = Nest_morphism(domain_g, codomain_g, map_g)
    return f, g


def random_mutually_refinable_nested_tuples():
    """
    Generate a pair of mutually refinable nested tuples. 
    
    :return: Pair of mutually refinable nested tuples
    :rtype: Tuple[NestedTuple, NestedTuple]
    """
    MAX_TOTAL = 2**31 - 1
    
    # Build list1 with product <= MAX_TOTAL
    length1 = np.random.randint(1, 10)
    list1 = []
    size1 = 1
    
    for _ in range(length1):
        max_factor = MAX_TOTAL // size1
        if max_factor < 2:
            break
        max_power = min(5, max_factor.bit_length() - 1)
        power = np.random.randint(1, max_power + 1)
        val = 2**power
        if size1 * val > MAX_TOTAL:
            break
        list1.append(val)
        size1 *= val
        
    if not list1:
        list1 = [2]
        size1 = 2

    # Build list2 so that prod(list2) == size1
    list2 = []
    size2 = 1
    
    while size2 < size1:
        max_factor = size1 // size2
        if max_factor < 2:
            break
        max_power = min(5, max_factor.bit_length() - 1)
        power = np.random.randint(1, max_power + 1)
        val = 2**power
        if size2 * val > size1:
            continue
        list2.append(val)
        size2 *= val

    tuple1 = tuple(list1)
    tuple2 = tuple(list2)
    profile1 = random_profile(length=len(tuple1), max_depth=3)
    profile2 = random_profile(length=len(tuple2), max_depth=3)
    nestedtuple1 = profile1.sub(tuple1)
    nestedtuple2 = profile2.sub(tuple2)
    
    return nestedtuple1, nestedtuple2


def random_weakly_composable_nest_morphisms():
    """
    Generate a pair of weakly composable nested tuple morphisms.
    
    :return: Pair of weakly composable morphisms
    :rtype: Tuple[Nest_morphism, Nest_morphism]
    """
    T, U = random_mutually_refinable_nested_tuples()
    f = random_Nest_morphism(codomain=T, nondegenerate=True)
    g = random_Nest_morphism(domain=U, nondegenerate=True)
    return f, g