"""
Category-theoretic morphisms implementation for Tract library.

This module implements morphisms in categories E_0, Tuple, and NestTuple,
providing the mathematical foundation for GPU kernel layout computations.
"""

from typing import Tuple, Optional

try:
    import cutlass
    import cutlass.cute as cute
except ImportError:
    # Allow import without cutlass for testing pure morphism operations
    cutlass = None
    cute = None


# *************************************************************************
# THE CATEGORY E_0 = E_0^otimes
# *************************************************************************


class Fin_morphism:
    """
    Morphisms in the category E_0 = E_0^otimes (finite pointed sets).
    
    A morphism α: <m>_* → <n>_* is encoded as:
    - domain: m (integer)
    - codomain: n (integer) 
    - map: tuple of length m where map[i-1] = 0 if α(i) = *, else α(i)
    
    :param domain: Size of the domain set
    :type domain: int
    :param codomain: Size of the codomain set
    :type codomain: int
    :param map_: Tuple encoding the morphism mapping
    :type map_: Tuple[int]
    :param name: Optional name for the morphism
    :type name: str
    """

    def __init__(self, domain: int, codomain: int, map_: Tuple[int], name: str = ""):
        self.domain = domain
        self.codomain = codomain
        self.map = map_
        self.name = name
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Validate that the input data defines a valid morphism in E_0.
        
        :raises ValueError: If the morphism is invalid
        """
        if len(self.map) != self.domain:
            raise ValueError(f"Map length ({len(self.map)}) must equal domain ({self.domain})")
            
        if not all(0 <= value <= self.codomain for value in self.map):
            raise ValueError(
                f"All values in the map must be between 0 and {self.codomain}"
            )

        nonzero_vals = [x for x in self.map if x > 0]
        if len(set(nonzero_vals)) < len(nonzero_vals):
            raise ValueError(f"The map ({self.map}) must contain no duplicate non-zero values")

    def __str__(self):
        """
        String representation of the morphism.
        
        :return: String representation
        :rtype: str
        """
        return f"<{self.domain}>* -{self.map}-> <{self.codomain}>*"

    def __repr__(self):
        return f"Fin_morphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"

    def compose(self, beta: "Fin_morphism") -> "Fin_morphism":
        """
        Compute the composition β ∘ α: <m>_* → <p>_*.
        
        :param beta: Second morphism to compose (must have domain = self.codomain)
        :type beta: Fin_morphism
        :return: The composition β ∘ α
        :rtype: Fin_morphism
        :raises ValueError: If morphisms are not composable
        """
        if self.codomain != beta.domain:
            raise ValueError("The given morphisms are not composable.")

        composite = []
        for value in self.map:
            if value > 0:
                composite.append(beta.map[value - 1])
            else:
                composite.append(0)
        
        return Fin_morphism(self.domain, beta.codomain, tuple(composite))

    def sum(self, beta: "Fin_morphism") -> "Fin_morphism":
        """
        Compute the sum α ⊕ β: <m+p>_* → <n+q>_*.
        
        :param beta: Second morphism in sum
        :type beta: Fin_morphism
        :return: The sum α ⊕ β
        :rtype: Fin_morphism
        """
        shifted = []
        for value in beta.map:
            if value == 0:
                shifted.append(0)
            else:
                shifted.append(value + self.codomain)
                
        return Fin_morphism(
            self.domain + beta.domain,
            self.codomain + beta.codomain,
            self.map + tuple(shifted),
        )

    def images_are_disjoint(self, beta: "Fin_morphism") -> bool:
        """
        Check if morphisms α and β have disjoint images.
        
        :param beta: Other morphism with same codomain
        :type beta: Fin_morphism
        :return: True if images are disjoint
        :rtype: bool
        :raises ValueError: If codomains differ
        """
        if self.codomain != beta.codomain:
            raise ValueError("The given maps do not have the same codomain.")

        # Construct the image of alpha
        seen_values = set()
        for value in self.map:
            if value > 0:
                seen_values.add(value)

        # Check that no value of beta is in the image of alpha
        for value in beta.map:
            if value > 0 and value in seen_values:
                return False

        return True

    def wedge(self, beta: "Fin_morphism") -> Optional["Fin_morphism"]:
        """
        Compute the wedge sum α ∨ β: <m+p>_* → <n>_*.
        
        Morphisms must have same codomain and disjoint images.
        
        :param beta: Second morphism in wedge sum
        :type beta: Fin_morphism
        :return: The wedge sum α ∨ β, or None if not valid
        :rtype: Optional[Fin_morphism]
        :raises ValueError: If morphisms don't have same codomain or disjoint images
        """
        if not self.codomain == beta.codomain:
            raise ValueError("The given morphisms do not have the same codomain.")
        if not self.images_are_disjoint(beta):
            raise ValueError("The given morphisms do not have disjoint images.")
            
        return Fin_morphism(
            self.domain + beta.domain, self.codomain, self.map + beta.map
        )


# *************************************************************************
# NESTED TUPLES
# *************************************************************************


class NestedTuple:
    """
    Nested tuple structure for hierarchical data representation.
    
    Supports arbitrarily nested tuples of integers with various operations
    for flattening, substitution, and refinement checking.
    
    :param data: Integer or nested tuple of integers
    :type data: int or tuple
    """
    
    def __init__(self, data):
        if not self._validate(data):
            raise ValueError("Only integers or nested tuples of integers are allowed.")
        self.data = data

    def _validate(self, obj):
        """
        Recursively validate the nested structure.
        
        :param obj: Object to validate
        :return: True if valid
        :rtype: bool
        """
        if isinstance(obj, int):
            return True
        elif isinstance(obj, tuple):
            return all(self._validate(item) for item in obj)
        return False

    def _custom_repr(self, obj):
        """
        Custom representation without trailing commas for single elements.
        
        :param obj: Object to represent
        :return: String representation
        :rtype: str
        """
        if isinstance(obj, int):
            return str(obj)
        elif isinstance(obj, tuple):
            if not obj:
                return "()"
            elif len(obj) == 1:
                return f"({self._custom_repr(obj[0])})"
            else:
                inner = ",".join(self._custom_repr(item) for item in obj)
                return f"({inner})"

    def __repr__(self):
        return self._custom_repr(self.data)

    def __str__(self):
        return repr(self)

    def _flatten(self, obj):
        """
        Generator for flattening nested structure.
        
        :param obj: Object to flatten
        :yield: Integer values
        """
        if isinstance(obj, int):
            yield obj
        elif isinstance(obj, tuple):
            for item in obj:
                yield from self._flatten(item)

    def flatten(self) -> tuple:
        """
        Return flattened tuple of all integers.
        
        :return: Flattened tuple
        :rtype: tuple
        """
        return tuple(self._flatten(self.data))

    def __iter__(self):
        return iter(self.flatten())

    def __getitem__(self, index):
        return self.flatten()[index]

    def length(self) -> int:
        """
        Number of integers in flattened representation.
        
        :return: Length of flattened tuple
        :rtype: int
        """
        return len(self.flatten())

    def rank(self) -> int:
        """
        Number of top-level modes.
        
        :return: Rank of the nested tuple
        :rtype: int
        """
        if isinstance(self.data, int):
            return 1
        return len(self.data)

    def size(self) -> int:
        """
        Product of all integers in the nested tuple.
        
        :return: Product of all entries
        :rtype: int
        """
        size = 1
        for entry in self.flatten():
            size *= entry
        return size

    def entry(self, i: int):
        """
        Get i-th entry (1-indexed).
        
        :param i: Index (1-based)
        :type i: int
        :return: Value at index i
        :rtype: int
        :raises IndexError: If index out of range
        """
        if i < 1 or i > self.length():
            raise IndexError("Index out of range")
        return self[i - 1]

    def mode(self, i: int) -> "NestedTuple":
        """
        Get i-th mode as a NestedTuple (1-indexed).
        
        :param i: Mode index (1-based)
        :type i: int
        :return: i-th mode
        :rtype: NestedTuple
        :raises IndexError: If mode index out of range
        """
        if not isinstance(i, int) or i < 1:
            raise IndexError("Mode index must be a positive integer.")

        if isinstance(self.data, int):
            if i == 1:
                return NestedTuple(self.data)
            else:
                raise IndexError("An integer NestedTuple S has only one mode.")

        if i > self.rank():
            raise IndexError(f"Mode index {i} out of range.")

        return NestedTuple(self.data[i - 1])
    
    def depth(self) -> int:
        """
        Maximum depth of nesting in the nested tuple.
        An integer has depth 1, a flat tuple has depth 1,
        a nested tuple has depth 1 + max depth of its modes.
        
        :return: Maximum nesting depth
        :rtype: int
        """
        if isinstance(self.data, int):
            return 0
        
        max_depth = 0
        for i in range(1, self.rank() + 1):
            mode_depth = self.mode(i).depth()
            max_depth = max(max_depth, mode_depth)
        
        return max_depth + 1

    def sub(self, values: tuple) -> "NestedTuple":
        """
        Substitute values into the nested structure.
        
        :param values: Values to substitute
        :type values: tuple
        :return: New NestedTuple with substituted values
        :rtype: NestedTuple
        :raises ValueError: If values length doesn't match
        """
        if len(values) != self.length():
            raise ValueError("Replacement tuple must have same length as NestedTuple")

        it = iter(values)

        def _replace(obj):
            if isinstance(obj, int):
                return next(it)
            elif isinstance(obj, tuple):
                return tuple(_replace(item) for item in obj)

        new_data = _replace(self.data)
        return NestedTuple(new_data)

    def profile(self) -> "NestedTuple":
        """
        Return profile (structure with all zeros).
        
        :return: Profile of the nested tuple
        :rtype: NestedTuple
        """
        return self.sub(tuple([0] * self.length()))

    def is_congruent_to(self, other: "NestedTuple") -> bool:
        """
        Check if two NestedTuples have the same profile.
        
        :param other: Other NestedTuple to compare
        :type other: NestedTuple
        :return: True if congruent
        :rtype: bool
        """
        return self.profile().data == other.profile().data

    def replace_empty_tuples_with_one(self) -> "NestedTuple":
        """
        Replace all empty tuples with 1.
        
        :return: New NestedTuple with replacements
        :rtype: NestedTuple
        """
        def _replace(obj):
            if obj == ():
                return 1
            elif isinstance(obj, int):
                return obj
            elif isinstance(obj, tuple):
                return tuple(_replace(item) for item in obj)
            else:
                raise ValueError("Invalid element type.")

        new_data = _replace(self.data)
        return NestedTuple(new_data)

    def replace_empty_tuples_with_zero(self) -> "NestedTuple":
        """
        Replace all empty tuples with 0.
        
        :return: New NestedTuple with replacements
        :rtype: NestedTuple
        """
        def _replace(obj):
            if obj == ():
                return 0
            elif isinstance(obj, int):
                return obj
            elif isinstance(obj, tuple):
                return tuple(_replace(item) for item in obj)
            else:
                raise ValueError("Invalid element type.")

        new_data = _replace(self.data)
        return NestedTuple(new_data)

    def refines(self, other: "NestedTuple") -> bool:
        """
        Check if self refines other.
        
        S refines T if:
        1. S = T, or
        2. T = size(S), or  
        3. rank(S) = rank(T) and mode_i(S) refines mode_i(T) for all i
        
        :param other: NestedTuple to check refinement against
        :type other: NestedTuple
        :return: True if self refines other
        :rtype: bool
        """
        # Case 1: S = T
        if self.data == other.data:
            return True
        # Case 2: T = size(S)
        if isinstance(other.data, int) and other.data == self.size():
            return True
        # Case 3: rank(S) = rank(T) and mode_i(S) refines mode_i(T) for all i
        if self.rank() == other.rank():
            for i in range(1, self.rank() + 1):
                if not self.mode(i).refines(other.mode(i)):
                    return False
            return True
        return False

    def is_refined_by(self, other: "NestedTuple") -> bool:
        """
        Check if self is refined by other.
        
        :param other: NestedTuple to check
        :type other: NestedTuple
        :return: True if self is refined by other
        :rtype: bool
        """
        return other.refines(self)

    def relative_mode(self, i: int, other: "NestedTuple") -> "NestedTuple":
        """
        Get relative mode with respect to another NestedTuple.
        
        :param i: Index (1-based)
        :type i: int
        :param other: Reference NestedTuple
        :type other: NestedTuple
        :return: Relative mode
        :rtype: NestedTuple
        """
        assert self.refines(other), "Self must refine other"
        assert 1 <= i <= other.length()
        
        if i == 1 and other.data == self.size():
            return self
        else:
            l = 0
            N = 0
            while N + other.mode(l + 1).length() < i:
                l += 1
                N += other.mode(l).length()
            l += 1
            return self.mode(l).relative_mode(i - N, other.mode(l))

    def relative_flattening(self, other: "NestedTuple") -> "NestedTuple":
        """
        Compute relative flattening with respect to another NestedTuple.
        
        :param other: Reference NestedTuple
        :type other: NestedTuple
        :return: Relative flattening
        :rtype: NestedTuple
        """
        assert self.refines(other)
        result_list = []
        for i in range(1, other.length() + 1):
            result_list.append(self.relative_mode(i, other).data)
        return NestedTuple(tuple(result_list))

    def underlying_map(self, other: "NestedTuple") -> tuple:
        """
        Compute underlying map for refinement.
        
        :param other: Reference NestedTuple
        :type other: NestedTuple
        :return: Underlying map
        :rtype: tuple
        """
        assert self.refines(other)
        map_ = []
        for i in range(1, other.length() + 1):
            for _ in range(self.relative_mode(i, other).length()):
                map_.append(i)
        return tuple(map_)

    def sublength(self, i: int, refined: "NestedTuple") -> int:
        """
        Compute sublength up to position i.
        
        :param i: Position (1-based)
        :type i: int
        :param refined: Reference NestedTuple
        :type refined: NestedTuple
        :return: Sublength
        :rtype: int
        """
        assert 1 <= i <= self.length()
        result = 0
        for j in range(1, i):
            result += self.relative_mode(j, refined).length()
        return result

# *************************************************************************
# THE CATEGORY Tuple
# *************************************************************************


class Tuple_morphism:
    """
    Morphisms in the category Tuple.
    
    A morphism f: (s₁,...,sₘ) → (t₁,...,tₙ) lying over α: <m>_* → <n>_*
    is encoded with domain, codomain tuples and underlying map α.
    
    :param domain: Domain tuple
    :type domain: Tuple[int]
    :param codomain: Codomain tuple
    :type codomain: Tuple[int]
    :param map: Underlying Fin morphism map
    :type map: Tuple[int]
    :param name: Optional name
    :type name: str
    """

    def __init__(
        self, domain: Tuple[int], codomain: Tuple[int], map: Tuple[int], name: str = ""
    ):
        self.domain = domain
        self.codomain = codomain
        self.map = map
        self.name = name
        self.underlying_map = Fin_morphism(
            len(self.domain), len(self.codomain), self.map, self.name
        )
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Verify that the input data defines a valid morphism in Tuple category.
        
        :raises ValueError: If morphism is invalid
        """
        if len(self.domain) != self.underlying_map.domain:
            raise ValueError(
                f"Domain length ({len(self.domain)}) must match underlying map domain"
            )

        if len(self.codomain) != self.underlying_map.codomain:
            raise ValueError(
                f"Codomain length ({len(self.codomain)}) must match underlying map codomain"
            )

        for i, value in enumerate(self.underlying_map.map):
            if value != 0:
                if self.domain[i] != self.codomain[value - 1]:
                    raise ValueError(
                        f"Must satisfy s_i = t_α(i) for all i"
                    )

    def __repr__(self):
        return f"Tuple_morphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"

    def __str__(self):
        return f"{self.domain} --{self.map}--> {self.codomain}"

    def size(self) -> int:
        """
        Product of domain entries.
        
        :return: Size of domain
        :rtype: int
        """
        size = 1
        for entry in self.domain:
            size *= entry
        return size

    def cosize(self) -> int:
        """
        Product of codomain entries.
        
        :return: Size of codomain
        :rtype: int
        """
        cosize = 1
        for entry in self.codomain:
            cosize *= entry
        return cosize

    def is_sorted(self) -> bool:
        """
        Check if the morphism is sorted.
        
        :return: True if sorted
        :rtype: bool
        """
        m = len(self.domain)
        map_ = self.map
        
        for i, value in enumerate(map_):
            if (value == 0) and (i > 0):
                if (map_[i - 1] != 0) or (
                    (map_[i - 1] == 0) and self.domain[i - 1] > self.domain[i]
                ):
                    return False
                    
            if (i < m - 1) and (value != 0) and (map_[i + 1] != 0) and (value > map_[i + 1]):
                for j in range(map_[i + 1] - 1, value):
                    if self.codomain[j] != 1:
                        return False
        return True

    def is_coalesced(self) -> bool:
        """
        Check if the morphism is coalesced.
        
        :return: True if coalesced
        :rtype: bool
        """
        m = len(self.domain)
        map_ = self.map
        
        for i in range(m):
            if self.domain[i] == 1:
                return False
            if (i < m - 1) and (map_[i] > 0) and (map_[i] < map_[i + 1]):
                result = False
                for j in range(map_[i] + 1, map_[i + 1]):
                    if self.codomain[j - 1] > 1:
                        result = True
                if not result:
                    return False
        return True

    def are_composable(self, g: "Tuple_morphism") -> bool:
        """
        Check if morphisms are composable.
        
        :param g: Second morphism
        :type g: Tuple_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain == g.domain

    def compose(self, g: "Tuple_morphism") -> "Tuple_morphism":
        """
        Compute composition g ∘ f.
        
        :param g: Second morphism (must have domain = self.codomain)
        :type g: Tuple_morphism
        :return: The composition g ∘ f
        :rtype: Tuple_morphism
        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        return Tuple_morphism(
            self.domain, g.codomain, self.underlying_map.compose(g.underlying_map).map
        )

    def sum(self, g: "Tuple_morphism") -> "Tuple_morphism":
        """
        Compute sum f ⊕ g.
        
        :param g: Second morphism
        :type g: Tuple_morphism
        :return: Sum of morphisms
        :rtype: Tuple_morphism
        """
        return Tuple_morphism(
            self.domain + g.domain,
            self.codomain + g.codomain,
            self.underlying_map.sum(g.underlying_map).map,
        )

    def restrict(self, subtuple: tuple) -> "Tuple_morphism":
        """
        Restrict morphism to a subtuple of its domain.
        
        :param subtuple: Indices to restrict to (1-based)
        :type subtuple: tuple
        :return: Restricted morphism
        :rtype: Tuple_morphism
        :raises ValueError: If invalid subtuple
        """
        if not all(1 <= index <= len(self.domain) for index in subtuple):
            raise ValueError("Invalid subtuple indices.")

        if not all(subtuple[i] < subtuple[i + 1] for i in range(len(subtuple) - 1)):
            raise ValueError("Subtuple must be strictly increasing.")

        restricted_domain = tuple([self.domain[index - 1] for index in subtuple])
        restricted_map = tuple([self.map[index - 1] for index in subtuple])

        return Tuple_morphism(restricted_domain, self.codomain, restricted_map)

    def factorize(self, subtuple: Tuple[int]) -> "Tuple_morphism":
        """
        Factorize morphism through a subtuple of its codomain.
        
        :param subtuple: Codomain indices (1-based)
        :type subtuple: Tuple[int]
        :return: Factorized morphism
        :rtype: Tuple_morphism
        """
        domain = self.domain
        codomain = tuple([self.codomain[j - 1] for j in subtuple])
        map_ = []
        
        for value in self.map:
            if value == 0:
                map_.append(value)
            else:
                missing_count = sum(1 for i in range(1, value) if i not in subtuple)
                map_.append(value - missing_count)
                
        return Tuple_morphism(domain, codomain, tuple(map_))

    def sort(self) -> "Tuple_morphism":
        """
        Return sorted version of the morphism.
        
        :return: Sorted morphism
        :rtype: Tuple_morphism
        """
        alpha = self.map

        # Extract P (indices with α(i) = *) and Q (indices with α(i) ≠ *)
        P = [i + 1 for i, value in enumerate(alpha) if value == 0]
        Q = [i + 1 for i, value in enumerate(alpha) if value != 0]

        # Reorder P by domain values
        P_sorted = sorted(P, key=lambda i: (self.domain[i - 1], i))

        # Reorder Q by map values
        Q_sorted = sorted(Q, key=lambda j: alpha[j - 1])

        permutation = P_sorted + Q_sorted
        domain_of_g = [self.domain[entry - 1] for entry in permutation]

        g = Tuple_morphism(tuple(domain_of_g), self.domain, tuple(permutation))
        return g.compose(self)

    def images_are_disjoint(self, g: "Tuple_morphism") -> bool:
        """
        Check if morphisms have disjoint images.
        
        :param g: Other morphism
        :type g: Tuple_morphism
        :return: True if disjoint
        :rtype: bool
        """
        return self.underlying_map.images_are_disjoint(g.underlying_map)

    def concat(self, g: "Tuple_morphism") -> "Tuple_morphism":
        """
        Compute concatenation of morphisms with same codomain and disjoint images.
        
        :param g: Second morphism
        :type g: Tuple_morphism
        :return: Concatenation of f and g
        :rtype: Tuple_morphism
        :raises ValueError: If not valid for concatenation
        """
        if not self.images_are_disjoint(g):
            raise ValueError("Morphisms must have disjoint images.")

        wedge_result = self.underlying_map.wedge(g.underlying_map)
        if wedge_result is None:
            raise ValueError("The given morphisms do not have disjoint images.")
            
        concat = Tuple_morphism(self.domain + g.domain, self.codomain, wedge_result.map)
        
        if concat.cosize() > 2**31 - 1:
            raise ValueError("The cosize of the concatenation is too large.")
        
        return concat

    def squeeze(self) -> "Tuple_morphism":
        """
        Remove all ones from domain and codomain.
        
        :return: Squeezed morphism
        :rtype: Tuple_morphism
        """
        domain_subtuple = tuple(
            i + 1 for i in range(len(self.domain)) if self.domain[i] != 1
        )
        restricted_morphism = self.restrict(domain_subtuple)
        
        codomain_subtuple = tuple(
            j + 1 for j in range(len(restricted_morphism.codomain))
            if restricted_morphism.codomain[j] != 1
        )

        return restricted_morphism.factorize(codomain_subtuple)

    def strong_coalesce(self) -> "Tuple_morphism":
        """
        Compute strong coalescence of the morphism.
        
        :return: Strongly coalesced morphism
        :rtype: Tuple_morphism
        """
        morphism = self.squeeze()
        m = len(morphism.domain)
        n = len(morphism.codomain)

        # Form equivalence classes for domain
        if m > 0:
            domain_equivalence_classes = []
            current_equivalence_class = [1]
            for i in range(1, m):
                previous_value = morphism.map[i - 1]
                current_value = morphism.map[i]
                if (previous_value == 0 and current_value == 0) or (
                    (previous_value != 0) and current_value == previous_value + 1
                ):
                    current_equivalence_class.append(i + 1)
                else:
                    domain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [i + 1]
            domain_equivalence_classes.append(current_equivalence_class)
        else:
            domain_equivalence_classes = []

        # Form equivalence classes for codomain
        image_of_map = set(morphism.map)
        if n > 0:
            codomain_equivalence_classes = []
            current_equivalence_class = [1]
            for j in range(2, n + 1):
                if j - 1 not in image_of_map and j not in image_of_map:
                    current_equivalence_class.append(j)
                elif j - 1 in image_of_map:
                    i = 0
                    while morphism.map[i] != j - 1:
                        i += 1
                    if (i + 1 < m) and (morphism.map[i + 1] == j):
                        current_equivalence_class.append(j)
                    else:
                        codomain_equivalence_classes.append(current_equivalence_class)
                        current_equivalence_class = [j]
                else:
                    codomain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [j]
            codomain_equivalence_classes.append(current_equivalence_class)
        else:
            codomain_equivalence_classes = []
            
        # Build coalesced domain
        coalesced_domain = []
        for equivalence_class in domain_equivalence_classes:
            product = 1
            for index in equivalence_class:
                product *= morphism.domain[index - 1]
            coalesced_domain.append(product)
            
        # Build coalesced codomain
        coalesced_codomain = []
        for equivalence_class in codomain_equivalence_classes:
            product = 1
            for index in equivalence_class:
                product *= morphism.codomain[index - 1]
            coalesced_codomain.append(product)
            
        # Build coalesced map
        coalesced_map = []
        for i in range(len(coalesced_domain)):
            domain_representative = domain_equivalence_classes[i][0]
            codomain_representative = morphism.map[domain_representative - 1]
            if morphism.map[domain_representative - 1] == 0:
                coalesced_map.append(0)
            else:
                index = 0
                while (index < len(coalesced_codomain)) and (
                    codomain_representative not in codomain_equivalence_classes[index]
                ):
                    index += 1
                coalesced_map.append(index + 1)

        return Tuple_morphism(
            tuple(coalesced_domain), tuple(coalesced_codomain), tuple(coalesced_map)
        )

    def coalesce(self) -> "Tuple_morphism":
        """
        Compute weak coalescence of the morphism.
        
        :return: Weakly coalesced morphism
        :rtype: Tuple_morphism
        """
        morphism = self.squeeze()
        m = len(morphism.domain)
        n = len(morphism.codomain)

        # Form equivalence classes for domain
        if m > 0:
            domain_equivalence_classes = []
            current_equivalence_class = [1]
            for i in range(1, m):
                previous_value = morphism.map[i - 1]
                current_value = morphism.map[i]
                if (previous_value == 0 and current_value == 0) or (
                    (previous_value != 0) and current_value == previous_value + 1
                ):
                    current_equivalence_class.append(i + 1)
                else:
                    domain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [i + 1]
            domain_equivalence_classes.append(current_equivalence_class)
        else:
            domain_equivalence_classes = []

        # Form equivalence classes for codomain
        image_of_map = set(morphism.map)
        if n > 0:
            codomain_equivalence_classes = []
            current_equivalence_class = [1]
            for j in range(2, n + 1):
                if j - 1 in image_of_map:
                    i = 0
                    while morphism.map[i] != j - 1:
                        i += 1
                    if (i + 1 < m) and (morphism.map[i + 1] == j):
                        current_equivalence_class.append(j)
                    else:
                        codomain_equivalence_classes.append(current_equivalence_class)
                        current_equivalence_class = [j]
                else:
                    codomain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [j]
            codomain_equivalence_classes.append(current_equivalence_class)
        else:
            codomain_equivalence_classes = []
            
        # Build coalesced domain
        coalesced_domain = []
        for equivalence_class in domain_equivalence_classes:
            product = 1
            for index in equivalence_class:
                product *= morphism.domain[index - 1]
            coalesced_domain.append(product)
            
        # Build coalesced codomain  
        coalesced_codomain = []
        for equivalence_class in codomain_equivalence_classes:
            product = 1
            for index in equivalence_class:
                product *= morphism.codomain[index - 1]
            coalesced_codomain.append(product)
            
        # Build coalesced map
        coalesced_map = []
        for i in range(len(coalesced_domain)):
            domain_representative = domain_equivalence_classes[i][0]
            codomain_representative = morphism.map[domain_representative - 1]
            if morphism.map[domain_representative - 1] == 0:
                coalesced_map.append(0)
            else:
                index = 0
                while (index < len(coalesced_codomain)) and (
                    codomain_representative not in codomain_equivalence_classes[index]
                ):
                    index += 1
                coalesced_map.append(index + 1)

        return Tuple_morphism(
            tuple(coalesced_domain), tuple(coalesced_codomain), tuple(coalesced_map)
        )

    def coalesce_with_equiv(self):
        """
        Compute weak coalescence with equivalence classes.
        
        :return: Tuple of (coalesced morphism, codomain equivalence classes)
        :rtype: Tuple[Tuple_morphism, list]
        """
        # Implementation identical to coalesce() but returns equivalence classes
        morphism = self.squeeze()
        m = len(morphism.domain)
        n = len(morphism.codomain)

        # [Rest of implementation identical to coalesce()...]
        # Returning both morphism and equivalence classes
        
        # Form equivalence classes for domain
        if m > 0:
            domain_equivalence_classes = []
            current_equivalence_class = [1]
            for i in range(1, m):
                previous_value = morphism.map[i - 1]
                current_value = morphism.map[i]
                if (previous_value == 0 and current_value == 0) or (
                    (previous_value != 0) and current_value == previous_value + 1
                ):
                    current_equivalence_class.append(i + 1)
                else:
                    domain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [i + 1]
            domain_equivalence_classes.append(current_equivalence_class)
        else:
            domain_equivalence_classes = []

        # Form equivalence classes for codomain
        image_of_map = set(morphism.map)
        if n > 0:
            codomain_equivalence_classes = []
            current_equivalence_class = [1]
            for j in range(2, n + 1):
                if j - 1 in image_of_map:
                    i = 0
                    while morphism.map[i] != j - 1:
                        i += 1
                    if (i + 1 < m) and (morphism.map[i + 1] == j):
                        current_equivalence_class.append(j)
                    else:
                        codomain_equivalence_classes.append(current_equivalence_class)
                        current_equivalence_class = [j]
                else:
                    codomain_equivalence_classes.append(current_equivalence_class)
                    current_equivalence_class = [j]
            codomain_equivalence_classes.append(current_equivalence_class)
        else:
            codomain_equivalence_classes = []
            
        coalesced_domain = []
        for equivalence_class in domain_equivalence_classes:
            product = 1
            for index in equivalence_class:
                product *= morphism.domain[index - 1]
            coalesced_domain.append(product)
            
        coalesced_codomain = []
        for equivalence_class in codomain_equivalence_classes:
            product = 1
            for index in equivalence_class:
                product *= morphism.codomain[index - 1]
            coalesced_codomain.append(product)
            
        coalesced_map = []
        for i in range(len(coalesced_domain)):
            domain_representative = domain_equivalence_classes[i][0]
            codomain_representative = morphism.map[domain_representative - 1]
            if morphism.map[domain_representative - 1] == 0:
                coalesced_map.append(0)
            else:
                index = 0
                while (index < len(coalesced_codomain)) and (
                    codomain_representative not in codomain_equivalence_classes[index]
                ):
                    index += 1
                coalesced_map.append(index + 1)

        morphism = Tuple_morphism(
            tuple(coalesced_domain), tuple(coalesced_codomain), tuple(coalesced_map)
        )
        
        return morphism, codomain_equivalence_classes

    def update_codomain(self, equivalence_relation: list) -> "Tuple_morphism":
        """
        Update codomain according to equivalence relation.
        
        :param equivalence_relation: List of equivalence classes
        :type equivalence_relation: list
        :return: Morphism with updated codomain
        :rtype: Tuple_morphism
        """
        new_codomain = []
        for class_ in equivalence_relation:
            new_entry = 1
            for representative in class_:
                new_entry *= self.codomain[representative - 1]
            new_codomain.append(new_entry)
        
        new_map = []
        for i in range(1, len(self.map) + 1):
            if self.map[i - 1] == 0:
                new_map.append(0)
            else:
                for j, class_ in enumerate(equivalence_relation):
                    if self.map[i - 1] in class_:
                        new_map.append(j + 1)
                        break
        
        return Tuple_morphism(self.domain, tuple(new_codomain), tuple(new_map))

    def is_complementable(self) -> bool:
        """
        Check if morphism is complementable.
        
        :return: True if complementable
        :rtype: bool
        """
        return 0 not in set(self.map)

    def complement(self) -> "Tuple_morphism":
        """
        Compute the complement of f.
        
        :return: Complement morphism
        :rtype: Tuple_morphism
        :raises ValueError: If not complementable
        """
        if not self.is_complementable():
            raise ValueError("The given morphism is not admissible for complementation.")

        codomain = self.codomain
        image_indices = set(self.map)
        domain = tuple(
            codomain[i] for i in range(len(codomain)) if i + 1 not in image_indices
        )
        map = tuple(i + 1 for i in range(len(codomain)) if i + 1 not in image_indices)
        
        return Tuple_morphism(domain, codomain, map)

    def is_isomorphism(self) -> bool:
        """
        Check if f is an isomorphism.
        
        :return: True if isomorphism
        :rtype: bool
        """
        m = len(self.domain)
        n = len(self.codomain)
        if (m == n) and set(self.map) == set(range(1, m + 1)):
            return True
        return False

    def is_complementary_to(self, other: "Tuple_morphism") -> bool:
        """
        Check if morphism is complementary to other morphism.
        
        :param other: Other morphism
        :type other: Tuple_morphism
        :return: True if complementary
        :rtype: bool
        :raises ValueError: If codomains differ
        """
        if self.codomain != other.codomain:
            raise ValueError("The given morphisms do not have the same codomain.")
        concat = self.concat(other)
        return concat.is_isomorphism()

    def to_Nest_morphism(self) -> "Nest_morphism":
        """
        Convert to nested tuple morphism.
        
        :return: Equivalent Nest_morphism
        :rtype: Nest_morphism
        """
        domain = NestedTuple(self.domain)
        codomain = NestedTuple(self.codomain)
        return Nest_morphism(domain, codomain, self.map)

    def flat_divide(self, other: "Tuple_morphism") -> "Tuple_morphism":
        """
        Compute flat division self / other.
        
        :param other: Denominator morphism
        :type other: Tuple_morphism
        :return: Quotient morphism
        :rtype: Tuple_morphism
        :raises ValueError: If division not valid
        """
        if not other.is_complementable():
            raise ValueError("The given denominator is not complementable.")
        if other.codomain != self.domain:
            raise ValueError("Codomain of denominator does not equal domain of numerator")

        return other.concat(other.complement()).compose(self)

    def flat_product(self, other: "Tuple_morphism") -> "Tuple_morphism":
        """
        Compute flat product self × other.
        
        :param other: Second factor
        :type other: Tuple_morphism
        :return: Product morphism
        :rtype: Tuple_morphism
        :raises ValueError: If product not valid
        """
        if not self.is_complementable():
            raise ValueError("The first factor is not complementable")
        if other.codomain != self.complement().domain:
            raise ValueError(
                "Domain of complement of first factor does not equal codomain of second factor"
            )

        return self.concat(other.compose(self.complement()))


# *************************************************************************
# THE CATEGORY NestTuple
# *************************************************************************


class Nest_morphism:
    """
    Morphisms in the category NestTuple.
    
    A morphism f: S → T between nested tuples lying over α: <m>_* → <n>_*.
    
    :param domain: Domain nested tuple
    :type domain: NestedTuple
    :param codomain: Codomain nested tuple
    :type codomain: NestedTuple
    :param map: Underlying map
    :type map: tuple
    :param name: Optional name
    :type name: str
    """

    def __init__(
        self, domain: NestedTuple | Tuple[int] | int, codomain: NestedTuple | Tuple[int] | int, map: tuple, name: str = ""
    ):
        self.domain = domain if isinstance(domain, NestedTuple) else NestedTuple(domain)
        self.codomain = codomain if isinstance(codomain, NestedTuple) else NestedTuple(codomain)
        self.map = map
        self.name = name
        self.underlying_map = Fin_morphism(
            len(self.domain.flatten()),
            len(self.codomain.flatten()),
            self.map,
            self.name,
        )
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """
        Verify that the input data defines a valid morphism.
        
        :raises ValueError: If morphism is invalid
        """
        if len(self.domain.flatten()) != self.underlying_map.domain:
            raise ValueError(
                f"Domain must match underlying map domain"
            )

        if len(self.codomain.flatten()) != self.underlying_map.codomain:
            raise ValueError(
                f"Codomain must match underlying map codomain"
            )

        for i, value in enumerate(self.underlying_map.map):
            if value != 0:
                if self.domain.flatten()[i] != self.codomain.flatten()[value - 1]:
                    raise ValueError(
                        f"Must satisfy s_i = t_α(i) for all i"
                    )

    def __repr__(self):
        return f"Nest_morphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"

    def __str__(self):
        return f"{self.domain} --{self.map}--> {self.codomain}"
    
    def repr_in_tex(self) -> str:
        map_str = '(' + ','.join(str(i) if i != 0 else '*' for i in self.map) + ')'
        return f"${self.domain} \\xrightarrow{{{map_str}}} {self.codomain}$"

    def flatten(self) -> Tuple_morphism:
        """
        Flatten to a Tuple_morphism.
        
        :return: Flattened morphism
        :rtype: Tuple_morphism
        """
        domain = self.domain.flatten()
        codomain = self.codomain.flatten()
        return Tuple_morphism(domain, codomain, self.map)

    def size(self) -> int:
        """
        Product of domain entries.
        
        :return: Size
        :rtype: int
        """
        size = 1
        for entry in self.domain.flatten():
            size *= entry
        return size

    def cosize(self) -> int:
        """
        Product of codomain entries.
        
        :return: Cosize
        :rtype: int
        """
        cosize = 1
        for entry in self.codomain.flatten():
            cosize *= entry
        return cosize

    def is_sorted(self) -> bool:
        """
        Check if the nested tuple morphism is sorted.
        
        :return: True if sorted
        :rtype: bool
        """
        return self.flatten().is_sorted()

    def are_composable(self, g: "Nest_morphism") -> bool:
        """
        Check if morphisms are composable.
        
        :param g: Second morphism
        :type g: Nest_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain.data == g.domain.data

    def compose(self, g: "Nest_morphism") -> "Nest_morphism":
        """
        Compute composition g ∘ f.
        
        :param g: Second morphism
        :type g: Nest_morphism
        :return: Composition
        :rtype: Nest_morphism
        :raises ValueError: If not composable
        """
        if self.codomain.data != g.domain.data:
            raise ValueError("The given morphisms are not composable.")

        return Nest_morphism(
            self.domain, g.codomain, self.underlying_map.compose(g.underlying_map).map
        )

    def images_are_disjoint(self, g: "Nest_morphism") -> bool:
        """
        Check if morphisms have disjoint images.
        
        :param g: Second morphism
        :type g: Nest_morphism
        :return: True if disjoint
        :rtype: bool
        :raises ValueError: If codomains differ
        """
        if self.codomain.data != g.codomain.data:
            raise ValueError("Morphisms do not have the same codomain.")
        return self.flatten().images_are_disjoint(g.flatten())

    def concat(self, g: "Nest_morphism") -> "Nest_morphism":
        """
        Compute concatenation (f,g) of nested tuple morphisms.
        
        :param g: Second morphism
        :type g: Nest_morphism
        :return: Concatenation
        :rtype: Nest_morphism
        :raises ValueError: If not valid for concatenation
        """
        if not self.images_are_disjoint(g):
            raise ValueError(
                "The given morphisms do not have the same codomain and disjoint images."
            )
        return Nest_morphism(
            NestedTuple((self.domain.data, g.domain.data)),
            self.codomain,
            self.underlying_map.wedge(g.underlying_map).map,
        )

    def coalesce(self) -> "Nest_morphism":
        """
        Compute coalescence of the morphism.

        :return: Coalesced morphism
        :rtype: Nest_morphism
        """
        flat_coalesce = self.flatten().coalesce().to_Nest_morphism()

        if flat_coalesce.domain.length() == 0:
            modification = Nest_morphism(1, (), (0,))
            result = modification.compose(flat_coalesce)
        elif flat_coalesce.domain.length() == 1:
            modification = Nest_morphism(flat_coalesce.domain.data[0], flat_coalesce.domain.data, (1,))
            result = modification.compose(flat_coalesce)
        else:
            result = flat_coalesce
        return result
        
    def is_complementable(self) -> bool:
        """
        Check if nested tuple morphism is complementable.
        
        :return: True if complementable
        :rtype: bool
        """
        return 0 not in set(self.map)

    def complement(self) -> "Nest_morphism":
        """
        Compute the complement of f.
        
        :return: Complement morphism
        :rtype: Nest_morphism
        :raises ValueError: If not complementable
        """
        if not self.is_complementable():
            raise ValueError("The given morphism is not complementable.")

        codomain = self.codomain
        image_indices = set(self.map)
        domain = [
            codomain[i] for i in range(codomain.length()) if i + 1 not in image_indices
        ]
        domain = NestedTuple(tuple(domain))
        map = tuple(
            i + 1 for i in range(codomain.length()) if i + 1 not in image_indices
        )

        return Nest_morphism(domain, codomain, map)

    def is_isomorphism(self) -> bool:
        """
        Check if f is an isomorphism.
        
        :return: True if isomorphism
        :rtype: bool
        """
        return self.flatten().is_isomorphism()

    def is_complementary_to(self, other: "Nest_morphism") -> bool:
        """
        Check if morphism is complementary to other morphism.
        
        :param other: Other morphism
        :type other: Nest_morphism
        :return: True if complementary
        :rtype: bool
        :raises ValueError: If codomains differ
        """
        if self.codomain.data != other.codomain.data:
            raise ValueError("The given morphisms do not have the same codomain.")

        concat = self.concat(other)
        return concat.is_isomorphism()

    def flatten_codomain(self) -> "Nest_morphism":
        """
        Flatten only the codomain.
        
        :return: Morphism with flattened codomain
        :rtype: Nest_morphism
        """
        domain = self.domain
        codomain = NestedTuple(self.codomain.flatten())
        return Nest_morphism(domain, codomain, self.map)

    def logical_divide(self, other: "Nest_morphism") -> "Nest_morphism":
        """
        Compute logical division.
        
        :param other: Denominator morphism
        :type other: Nest_morphism
        :return: Quotient
        :rtype: Nest_morphism
        """
        return other.concat(other.complement()).compose(self)

    def logical_product(self, other: "Nest_morphism") -> "Nest_morphism":
        """
        Compute logical product.
        
        :param other: Second factor
        :type other: Nest_morphism
        :return: Product
        :rtype: Nest_morphism
        """
        return self.concat(other.compose(self.complement()))

    def pullback_along(self, refinement: NestedTuple) -> "Nest_morphism":
        """
        Pullback morphism along a refinement.
        
        :param refinement: Refinement of codomain
        :type refinement: NestedTuple
        :return: Pullback morphism
        :rtype: Nest_morphism
        """
        S = self.domain
        T = self.codomain
        Tprime = refinement
        assert Tprime.refines(T)
        
        Sprime = []
        map_ = []
        for i in range(1, S.length() + 1):
            if self.map[i - 1] != 0:
                j = self.map[i - 1]
                Sprime.append(Tprime.relative_mode(j, T).data)
                for k in range(Tprime.relative_mode(j, T).length()):
                    map_.append(Tprime.sublength(j, T) + k + 1)
            else:
                Sprime.append(S.entry(i))
                map_.append(0)
                
        Sprime = S.sub(tuple(Sprime))
        map_ = tuple(map_)
        return Nest_morphism(Sprime, refinement, map_)

    def pushforward_along(self, refinement: NestedTuple) -> "Nest_morphism":
        """
        Pushforward morphism along a refinement.
        
        :param refinement: Refinement of domain
        :type refinement: NestedTuple
        :return: Pushforward morphism
        :rtype: Nest_morphism
        """
        U = self.domain
        V = self.codomain
        Uprime = refinement
        assert Uprime.refines(U)
        
        Vprime = []
        for j in range(1, V.length() + 1):
            if j not in set(self.map):
                Vprime.append(V.entry(j))
            else:
                i = self.map.index(j) + 1
                Vprime.append(Uprime.relative_mode(i, U).data)
                
        Vprime = V.sub(tuple(Vprime))
        
        map_ = []
        for i in range(1, U.length() + 1):
            if self.map[i - 1] == 0:
                for k in range(Uprime.relative_mode(i, U).length()):
                    map_.append(0)
            else:
                j = self.map[i - 1]
                for k in range(Vprime.relative_mode(j, V).length()):
                    map_.append(Vprime.sublength(j, V) + k + 1)
                    
        map_ = tuple(map_)
        return Nest_morphism(Uprime, Vprime, map_)

    def to_tikz(self, full_doc=False) -> str:
        """
        Generate TikZ representation of the morphism.

        :return: TikZ code
        :rtype: str
        """
        from tract.tuple_morph_tikz import nested_tuple_morphism_to_tikz

        return nested_tuple_morphism_to_tikz(
            self,
            row_spacing=0.8,
            tree_width=2.2,
            map_width=3.0,
            root_y_offset=0.0,
            label=f"{self.repr_in_tex()}",
            full_doc=full_doc,
        )

def make_morphism(domain, codomain, map, name="") -> Nest_morphism:
    """
    Create a Nest_morphism.
    
    :param domain: Domain nested tuple data
    :type domain: tuple
    :param codomain: Codomain nested tuple data
    :type codomain: tuple
    :param map: Underlying map
    :type map: tuple
    :param name: Optional name
    :type name: str
    :return: The created morphism
    :rtype: Nest_morphism
    """
    return Nest_morphism(NestedTuple(domain), NestedTuple(codomain), map, name)

def compose(f: Nest_morphism, g: Nest_morphism) -> Nest_morphism:
    return f.compose(g)

def coalesce(f: Nest_morphism) -> Nest_morphism:
    return f.coalesce()

def complement(f: Nest_morphism) -> Nest_morphism:
    return f.complement()

def logical_divide(f: Nest_morphism, g: Nest_morphism) -> Nest_morphism:
    return f.logical_divide(g)

def logical_product(f: Nest_morphism, g: Nest_morphism) -> Nest_morphism:
    return f.logical_product(g)

def morphism_to_tikz(f: Nest_morphism, full_doc=False) -> str:
    return f.to_tikz(full_doc=full_doc)