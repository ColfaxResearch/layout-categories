"""The category Nest of nested tuples, encoding layouts."""

from typing import Tuple, Optional

from .fin import FinMorphism
from .nested_tuple import NestedTuple
from .tuple_morphism import TupleMorphism
# *************************************************************************
# THE CATEGORY NestTuple
# *************************************************************************


class NestMorphism:
    """
    Morphisms in the category NestTuple.

    A morphism f: S → T between nested tuples lying over α: <m>_* → <n>_*.
    """

    def __init__(
        self, domain: NestedTuple | Tuple[int] | int, codomain: NestedTuple | Tuple[int] | int, map: tuple, name: str = ""
    ):
        self.domain = domain if isinstance(domain, NestedTuple) else NestedTuple(domain)
        self.codomain = codomain if isinstance(codomain, NestedTuple) else NestedTuple(codomain)
        self.map = map
        self.name = name
        self.underlying_map = FinMorphism(
            len(self.domain.flatten()),
            len(self.codomain.flatten()),
            self.map,
            self.name,
        )
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Verify that the input data defines a valid morphism."""
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
        return f"NestMorphism(domain={self.domain}, codomain={self.codomain}, map={self.map})"

    def __str__(self):
        return f"{self.domain} --{self.map}--> {self.codomain}"
    
    def repr_in_tex(self) -> str:
        map_str = '(' + ','.join(str(i) if i != 0 else '*' for i in self.map) + ')'
        return f"${self.domain} \\xrightarrow{{{map_str}}} {self.codomain}$"

    def flatten(self) -> TupleMorphism:
        """Flatten to a TupleMorphism."""
        domain = self.domain.flatten()
        codomain = self.codomain.flatten()
        return TupleMorphism(domain, codomain, self.map)

    def size(self) -> int:
        """Product of domain entries."""
        size = 1
        for entry in self.domain.flatten():
            size *= entry
        return size

    def cosize(self) -> int:
        """Product of codomain entries."""
        cosize = 1
        for entry in self.codomain.flatten():
            cosize *= entry
        return cosize

    def is_sorted(self) -> bool:
        """Check if the nested tuple morphism is sorted."""
        return self.flatten().is_sorted()

    def are_composable(self, g: "NestMorphism") -> bool:
        """Check if morphisms are composable."""
        return self.codomain.data == g.domain.data

    def __eq__(self, other):
        """Structural equality on (domain, codomain, map); names are ignored."""
        if not isinstance(other, NestMorphism):
            return NotImplemented
        return (
            self.domain == other.domain
            and self.codomain == other.codomain
            and self.map == other.map
        )

    def __hash__(self):
        return hash((self.domain, self.codomain, self.map))

    @classmethod
    def identity(cls, codomain: NestedTuple, name: str = "") -> "NestMorphism":
        """Identity morphism on a nested tuple."""
        return cls(
            codomain, codomain, tuple(range(1, codomain.length() + 1)), name
        )

    def is_identity(self) -> bool:
        """Check whether the morphism is an identity."""
        return self.domain == self.codomain and self.map == tuple(
            range(1, self.domain.length() + 1)
        )

    def compose(self, g: "NestMorphism") -> "NestMorphism":
        """Compute composition g ∘ f."""
        if self.codomain.data != g.domain.data:
            raise ValueError("The given morphisms are not composable.")

        return NestMorphism(
            self.domain, g.codomain, self.underlying_map.compose(g.underlying_map).map
        )

    def images_are_disjoint(self, g: "NestMorphism") -> bool:
        """Check if morphisms with the same codomain have disjoint images."""
        if self.codomain.data != g.codomain.data:
            raise ValueError("Morphisms do not have the same codomain.")
        return self.flatten().images_are_disjoint(g.flatten())

    def concat(self, g: "NestMorphism") -> "NestMorphism":
        """
        Compute concatenation (f,g) of nested tuple morphisms.

        Requires the same codomain and disjoint images.
        """
        if not self.images_are_disjoint(g):
            raise ValueError(
                "The given morphisms do not have the same codomain and disjoint images."
            )
        return NestMorphism(
            NestedTuple((self.domain.data, g.domain.data)),
            self.codomain,
            self.underlying_map.wedge(g.underlying_map).map,
        )

    def coalesce(self) -> "NestMorphism":
        """Compute coalescence of the morphism."""
        flat_coalesce = self.flatten().coalesce().to_nest_morphism()

        if flat_coalesce.domain.length() == 0:
            modification = NestMorphism(1, (), (0,))
            result = modification.compose(flat_coalesce)
        elif flat_coalesce.domain.length() == 1:
            modification = NestMorphism(flat_coalesce.domain.data[0], flat_coalesce.domain.data, (1,))
            result = modification.compose(flat_coalesce)
        else:
            result = flat_coalesce
        return result
        
    def is_complementable(self) -> bool:
        """Check if nested tuple morphism is complementable."""
        return 0 not in set(self.map)

    def complement(self) -> "NestMorphism":
        """Compute the complement of f."""
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

        return NestMorphism(domain, codomain, map)

    def is_isomorphism(self) -> bool:
        """Check if f is an isomorphism."""
        return self.flatten().is_isomorphism()

    def is_complementary_to(self, other: "NestMorphism") -> bool:
        """Check if morphism is complementary to other morphism."""
        if self.codomain.data != other.codomain.data:
            raise ValueError("The given morphisms do not have the same codomain.")

        concat = self.concat(other)
        return concat.is_isomorphism()

    def flatten_codomain(self) -> "NestMorphism":
        """Flatten only the codomain."""
        domain = self.domain
        codomain = NestedTuple(self.codomain.flatten())
        return NestMorphism(domain, codomain, self.map)

    def logical_divide(self, other: "NestMorphism") -> "NestMorphism":
        """Compute logical division by other."""
        return other.concat(other.complement()).compose(self)

    def logical_product(self, other: "NestMorphism") -> "NestMorphism":
        """Compute logical product with other."""
        return self.concat(other.compose(self.complement()))

    def pullback_along(self, refinement: NestedTuple) -> "NestMorphism":
        """Pullback morphism along a refinement of the codomain."""
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
        return NestMorphism(Sprime, refinement, map_)

    def pushforward_along(self, refinement: NestedTuple) -> "NestMorphism":
        """Pushforward morphism along a refinement of the domain."""
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
        return NestMorphism(Uprime, Vprime, map_)

    def to_tikz(self, full_doc=False) -> str:
        """Generate TikZ representation of the morphism."""
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

def make_morphism(domain, codomain, map, name="") -> NestMorphism:
    """Create a NestMorphism from raw nested tuple data."""
    return NestMorphism(NestedTuple(domain), NestedTuple(codomain), map, name)

def compose(f: NestMorphism, g: NestMorphism) -> NestMorphism:
    return f.compose(g)

def coalesce(f: NestMorphism) -> NestMorphism:
    return f.coalesce()

def complement(f: NestMorphism) -> NestMorphism:
    return f.complement()

def logical_divide(f: NestMorphism, g: NestMorphism) -> NestMorphism:
    return f.logical_divide(g)

def logical_product(f: NestMorphism, g: NestMorphism) -> NestMorphism:
    return f.logical_product(g)

def morphism_to_tikz(f: NestMorphism, full_doc=False) -> str:
    return f.to_tikz(full_doc=full_doc)