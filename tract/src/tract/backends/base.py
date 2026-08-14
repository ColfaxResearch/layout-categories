"""
Backend-independent layout algorithms, stated on plain (shape, stride)
tuples. The concrete backends wrap these in their own layout types.
"""

from ..tuple_morphism import TupleMorphism


def flatten_nested(x) -> tuple:
    """Flatten an int or arbitrarily nested tuple to a flat tuple of leaves."""
    if isinstance(x, tuple):
        return tuple(leaf for entry in x for leaf in flatten_nested(entry))
    return (x,)


def sorted_modes(shape: tuple, stride: tuple):
    """Sort flat (shape, stride) modes by stride, breaking ties by shape."""
    if len(shape) == 0:
        return shape, stride
    pairs = sorted(zip(shape, stride), key=lambda x: (x[1], x[0]))
    sorted_shape, sorted_stride = zip(*pairs)
    return tuple(sorted_shape), tuple(sorted_stride)


def sorted_modes_with_perm(shape: tuple, stride: tuple):
    """Sort flat modes by stride and return the 1-indexed permutation used."""
    if len(shape) == 0:
        return shape, stride, []
    indexed = list(enumerate(zip(shape, stride)))
    sorted_indexed = sorted(indexed, key=lambda x: (x[1][1], x[1][0]))
    permutation = [index + 1 for index, _ in sorted_indexed]
    sorted_shape, sorted_stride = zip(*[item for _, item in sorted_indexed])
    return tuple(sorted_shape), tuple(sorted_stride), permutation


def is_tractable_modes(shape: tuple, stride: tuple) -> bool:
    """
    Check tractability of a flat layout given as (shape, stride) tuples.

    A flat layout is tractable if, after sorting modes by stride, each
    nonzero stride divides the next stride-times-shape product — i.e. the
    prefix products form a divisibility chain, which happens exactly when
    the layout is the layout of a tuple morphism.
    """
    shape, stride = sorted_modes(flatten_nested(shape), flatten_nested(stride))
    for i in range(len(shape) - 1):
        if stride[i] != 0:
            if stride[i + 1] % (shape[i] * stride[i]) != 0:
                return False
    return True


def flat_layout_components(
    morphism: TupleMorphism, check_overflow: bool = False
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Return the (shape, stride) pair of the layout L_f associated to a tuple
    morphism f: mode i has shape domain[i] and stride the prefix product of
    the codomain below map[i] (0 for basepoint modes).

    :param check_overflow: Raise OverflowError past the 32-bit limit
        (required by the CuTe DSL backend).
    """
    domain = morphism.domain
    codomain = morphism.codomain
    alpha = morphism.map

    m = len(domain)
    stride_list = [0] * m

    for i in range(m):
        if alpha[i] != 0:
            t = 1
            for j in range(alpha[i] - 1):
                t *= codomain[j]
            if check_overflow and t > 2**31 - 1:
                raise OverflowError("Stride value exceeds 32-bit integer limit.")
            stride_list[i] = t

    return tuple(domain), tuple(stride_list)


def standard_tuple_morphism(shape: tuple, stride: tuple) -> TupleMorphism:
    """
    The standard representation f_L of a tractable flat layout L, given as
    flat (shape, stride) tuples.

    :raises ValueError: If the layout is not tractable
    """
    if not is_tractable_modes(shape, stride):
        raise ValueError("The provided layout is not tractable.")

    domain = tuple(shape)
    sorted_shape, sorted_stride, permutation = sorted_modes_with_perm(shape, stride)
    m = len(sorted_shape)

    # Find the largest integer k such that stride[k-1] = 0
    k = 0
    for entry in sorted_stride:
        if entry == 0:
            k += 1
        else:
            break

    # Build codomain
    codomain = tuple()
    if k < m:
        cod = [sorted_stride[k], sorted_shape[k]]
        for j in range(k + 1, m):
            denom = sorted_shape[j - 1] * sorted_stride[j - 1]
            factor = (sorted_stride[j] // denom) if denom != 0 else 0
            cod.append(int(factor))
            cod.append(sorted_shape[j])
        codomain = tuple(cod)

    # Construct the map alpha'
    alpha_prime = [0] * m
    for j in range(k, m):
        alpha_prime[j] = 2 * (j - k + 1)

    # Construct the inverse permutation
    inverse_permutation = [0] * m
    for i in range(m):
        inverse_permutation[permutation[i] - 1] = i + 1

    # alpha = alpha'[σ^{-1}(i)]
    alpha = tuple(alpha_prime[inverse_permutation[i] - 1] for i in range(m))

    morphism = TupleMorphism(domain, codomain, alpha)
    restricted_codomain_indices = []
    for i in range(len(codomain)):
        if codomain[i] != 1 or i + 1 in morphism.map:
            restricted_codomain_indices.append(i + 1)
    return morphism.factorize(tuple(restricted_codomain_indices))


def flat_complement_components(shape: tuple, stride: tuple, N: int):
    """
    The (shape, stride) pair of the complement of a flat layout with respect
    to total size N: the layout filling the gaps of the sorted input, so
    that their concatenation is a compact layout of size N.
    """
    S, D = sorted_modes(shape, stride)
    m = len(S)

    out_shape = [D[0]]
    for i in range(1, m):
        out_shape.append(D[i] // (S[i - 1] * D[i - 1]))
    out_shape.append(N // (S[-1] * D[-1]))

    out_stride = [1]
    for i in range(m):
        out_stride.append(S[i] * D[i])

    return tuple(out_shape), tuple(out_stride)
