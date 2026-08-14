"""The category E_0 = E_0^otimes of pointed finite sets."""

from typing import Tuple, Optional
# *************************************************************************
# THE CATEGORY E_0 = E_0^otimes
# *************************************************************************


class FinMorphism:
    """
    Morphisms in the category E_0 = E_0^otimes (finite pointed sets).

    A morphism α: <m>_* → <n>_* is encoded as:
    - domain: m (integer)
    - codomain: n (integer)
    - map: tuple of length m where map[i-1] = 0 if α(i) = *, else α(i)
    """

    def __init__(self, domain: int, codomain: int, map: Tuple[int], name: str = ""):
        self.domain = domain
        self.codomain = codomain
        self.map = map
        self.name = name
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate that the input data defines a valid morphism in E_0."""
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
        """String representation of the morphism."""
        return f"<{self.domain}>* -{self.map}-> <{self.codomain}>*"

    def __repr__(self):
        return f"FinMorphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"

    def __eq__(self, other):
        """Structural equality on (domain, codomain, map); names are ignored."""
        if not isinstance(other, FinMorphism):
            return NotImplemented
        return (
            self.domain == other.domain
            and self.codomain == other.codomain
            and self.map == other.map
        )

    def __hash__(self):
        return hash((self.domain, self.codomain, self.map))

    @classmethod
    def identity(cls, codomain: int, name: str = "") -> "FinMorphism":
        """Identity morphism on <codomain>."""
        return cls(codomain, codomain, tuple(range(1, codomain + 1)), name)

    def is_identity(self) -> bool:
        """Check whether the morphism is an identity."""
        return self.domain == self.codomain and self.map == tuple(
            range(1, self.domain + 1)
        )

    def compose(self, beta: "FinMorphism") -> "FinMorphism":
        """Compute the composition β ∘ α: <m>_* → <p>_*."""
        if self.codomain != beta.domain:
            raise ValueError("The given morphisms are not composable.")

        composite = []
        for value in self.map:
            if value > 0:
                composite.append(beta.map[value - 1])
            else:
                composite.append(0)
        
        return FinMorphism(self.domain, beta.codomain, tuple(composite))

    def sum(self, beta: "FinMorphism") -> "FinMorphism":
        """Compute the sum α ⊕ β: <m+p>_* → <n+q>_*."""
        shifted = []
        for value in beta.map:
            if value == 0:
                shifted.append(0)
            else:
                shifted.append(value + self.codomain)
                
        return FinMorphism(
            self.domain + beta.domain,
            self.codomain + beta.codomain,
            self.map + tuple(shifted),
        )

    def images_are_disjoint(self, beta: "FinMorphism") -> bool:
        """Check if morphisms α and β (with the same codomain) have disjoint images."""
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

    def wedge(self, beta: "FinMorphism") -> Optional["FinMorphism"]:
        """
        Compute the wedge sum α ∨ β: <m+p>_* → <n>_*.

        The morphisms must have the same codomain and disjoint images.
        """
        if not self.codomain == beta.codomain:
            raise ValueError("The given morphisms do not have the same codomain.")
        if not self.images_are_disjoint(beta):
            raise ValueError("The given morphisms do not have disjoint images.")
            
        return FinMorphism(
            self.domain + beta.domain, self.codomain, self.map + beta.map
        )


