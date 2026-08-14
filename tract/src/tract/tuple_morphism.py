"""The category Tuple of flat tuples, encoding flat layouts."""

from typing import Tuple, Optional

from .fin import FinMorphism
from .nested_tuple import NestedTuple
# *************************************************************************
# THE CATEGORY Tuple
# *************************************************************************


class TupleMorphism:
    """
    Morphisms in the category Tuple.

    A morphism f: (s₁,...,sₘ) → (t₁,...,tₙ) lying over α: <m>_* → <n>_*
    is encoded with domain, codomain tuples and underlying map α.
    """

    def __init__(
        self, domain: Tuple[int], codomain: Tuple[int], map: Tuple[int], name: str = ""
    ):
        self.domain = domain
        self.codomain = codomain
        self.map = map
        self.name = name
        self.underlying_map = FinMorphism(
            len(self.domain), len(self.codomain), self.map, self.name
        )
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Verify that the input data defines a valid morphism in Tuple: s_i = t_α(i) for all i."""
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
        return f"TupleMorphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"

    def __str__(self):
        return f"{self.domain} --{self.map}--> {self.codomain}"

    def size(self) -> int:
        """Product of domain entries."""
        size = 1
        for entry in self.domain:
            size *= entry
        return size

    def cosize(self) -> int:
        """Product of codomain entries."""
        cosize = 1
        for entry in self.codomain:
            cosize *= entry
        return cosize

    def is_sorted(self) -> bool:
        """Check if the morphism is sorted."""
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
        """Check if the morphism is coalesced."""
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

    def are_composable(self, g: "TupleMorphism") -> bool:
        """Check if morphisms are composable."""
        return self.codomain == g.domain

    def __eq__(self, other):
        """Structural equality on (domain, codomain, map); names are ignored."""
        if not isinstance(other, TupleMorphism):
            return NotImplemented
        return (
            self.domain == other.domain
            and self.codomain == other.codomain
            and self.map == other.map
        )

    def __hash__(self):
        return hash((self.domain, self.codomain, self.map))

    @classmethod
    def identity(cls, codomain: Tuple[int], name: str = "") -> "TupleMorphism":
        """Identity morphism on a flat tuple."""
        return cls(codomain, codomain, tuple(range(1, len(codomain) + 1)), name)

    def is_identity(self) -> bool:
        """Check whether the morphism is an identity."""
        return self.domain == self.codomain and self.map == tuple(
            range(1, len(self.domain) + 1)
        )

    def compose(self, g: "TupleMorphism") -> "TupleMorphism":
        """Compute composition g ∘ f."""
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        return TupleMorphism(
            self.domain, g.codomain, self.underlying_map.compose(g.underlying_map).map
        )

    def sum(self, g: "TupleMorphism") -> "TupleMorphism":
        """Compute sum f ⊕ g."""
        return TupleMorphism(
            self.domain + g.domain,
            self.codomain + g.codomain,
            self.underlying_map.sum(g.underlying_map).map,
        )

    def restrict(self, subtuple: tuple) -> "TupleMorphism":
        """Restrict morphism to a subtuple of its domain, given as strictly increasing 1-based indices."""
        if not all(1 <= index <= len(self.domain) for index in subtuple):
            raise ValueError("Invalid subtuple indices.")

        if not all(subtuple[i] < subtuple[i + 1] for i in range(len(subtuple) - 1)):
            raise ValueError("Subtuple must be strictly increasing.")

        restricted_domain = tuple([self.domain[index - 1] for index in subtuple])
        restricted_map = tuple([self.map[index - 1] for index in subtuple])

        return TupleMorphism(restricted_domain, self.codomain, restricted_map)

    def factorize(self, subtuple: Tuple[int]) -> "TupleMorphism":
        """Factorize morphism through a subtuple of its codomain, given as 1-based indices."""
        domain = self.domain
        codomain = tuple([self.codomain[j - 1] for j in subtuple])
        map_ = []
        
        for value in self.map:
            if value == 0:
                map_.append(value)
            else:
                missing_count = sum(1 for i in range(1, value) if i not in subtuple)
                map_.append(value - missing_count)
                
        return TupleMorphism(domain, codomain, tuple(map_))

    def sort(self) -> "TupleMorphism":
        """Return sorted version of the morphism."""
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

        g = TupleMorphism(tuple(domain_of_g), self.domain, tuple(permutation))
        return g.compose(self)

    def images_are_disjoint(self, g: "TupleMorphism") -> bool:
        """Check if morphisms have disjoint images."""
        return self.underlying_map.images_are_disjoint(g.underlying_map)

    def concat(self, g: "TupleMorphism") -> "TupleMorphism":
        """
        Compute concatenation of morphisms with same codomain and disjoint images.

        Raises ValueError if the cosize of the concatenation exceeds 2^31 - 1.
        """
        if not self.images_are_disjoint(g):
            raise ValueError("Morphisms must have disjoint images.")

        wedge_result = self.underlying_map.wedge(g.underlying_map)
        if wedge_result is None:
            raise ValueError("The given morphisms do not have disjoint images.")
            
        concat = TupleMorphism(self.domain + g.domain, self.codomain, wedge_result.map)
        
        if concat.cosize() > 2**31 - 1:
            raise ValueError("The cosize of the concatenation is too large.")
        
        return concat

    def squeeze(self) -> "TupleMorphism":
        """Remove all ones from domain and codomain."""
        domain_subtuple = tuple(
            i + 1 for i in range(len(self.domain)) if self.domain[i] != 1
        )
        restricted_morphism = self.restrict(domain_subtuple)
        
        codomain_subtuple = tuple(
            j + 1 for j in range(len(restricted_morphism.codomain))
            if restricted_morphism.codomain[j] != 1
        )

        return restricted_morphism.factorize(codomain_subtuple)

    def strong_coalesce(self) -> "TupleMorphism":
        """Compute strong coalescence of the morphism."""
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

        return TupleMorphism(
            tuple(coalesced_domain), tuple(coalesced_codomain), tuple(coalesced_map)
        )

    def coalesce(self) -> "TupleMorphism":
        """Compute weak coalescence of the morphism."""
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

        return TupleMorphism(
            tuple(coalesced_domain), tuple(coalesced_codomain), tuple(coalesced_map)
        )

    def coalesce_with_equiv(self):
        """Compute weak coalescence, returning (coalesced morphism, codomain equivalence classes)."""
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

        morphism = TupleMorphism(
            tuple(coalesced_domain), tuple(coalesced_codomain), tuple(coalesced_map)
        )
        
        return morphism, codomain_equivalence_classes

    def update_codomain(self, equivalence_relation: list) -> "TupleMorphism":
        """Update codomain according to an equivalence relation given as a list of equivalence classes."""
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
        
        return TupleMorphism(self.domain, tuple(new_codomain), tuple(new_map))

    def is_complementable(self) -> bool:
        """Check if morphism is complementable, i.e. its underlying map has no zero entries."""
        return 0 not in set(self.map)

    def complement(self) -> "TupleMorphism":
        """Compute the complement of f."""
        if not self.is_complementable():
            raise ValueError("The given morphism is not admissible for complementation.")

        codomain = self.codomain
        image_indices = set(self.map)
        domain = tuple(
            codomain[i] for i in range(len(codomain)) if i + 1 not in image_indices
        )
        map = tuple(i + 1 for i in range(len(codomain)) if i + 1 not in image_indices)
        
        return TupleMorphism(domain, codomain, map)

    def is_isomorphism(self) -> bool:
        """Check if f is an isomorphism."""
        m = len(self.domain)
        n = len(self.codomain)
        if (m == n) and set(self.map) == set(range(1, m + 1)):
            return True
        return False

    def is_complementary_to(self, other: "TupleMorphism") -> bool:
        """Check if morphism is complementary to other morphism with the same codomain."""
        if self.codomain != other.codomain:
            raise ValueError("The given morphisms do not have the same codomain.")
        concat = self.concat(other)
        return concat.is_isomorphism()

    def to_nest_morphism(self) -> "NestMorphism":
        """Convert to nested tuple morphism."""
        from .nest_morphism import NestMorphism

        domain = NestedTuple(self.domain)
        codomain = NestedTuple(self.codomain)
        return NestMorphism(domain, codomain, self.map)

    # Deprecated alias
    to_Nest_morphism = to_nest_morphism

    def flat_divide(self, other: "TupleMorphism") -> "TupleMorphism":
        """
        Compute flat division self / other.

        The denominator must be complementable, with codomain equal to
        the domain of the numerator.
        """
        if not other.is_complementable():
            raise ValueError("The given denominator is not complementable.")
        if other.codomain != self.domain:
            raise ValueError("Codomain of denominator does not equal domain of numerator")

        return other.concat(other.complement()).compose(self)

    def flat_product(self, other: "TupleMorphism") -> "TupleMorphism":
        """
        Compute flat product self × other.

        The first factor must be complementable, and the codomain of the
        second factor must equal the domain of the first factor's complement.
        """
        if not self.is_complementable():
            raise ValueError("The first factor is not complementable")
        if other.codomain != self.complement().domain:
            raise ValueError(
                "Domain of complement of first factor does not equal codomain of second factor"
            )

        return self.concat(other.compose(self.complement()))


