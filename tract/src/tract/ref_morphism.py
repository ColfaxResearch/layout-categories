"""
The category Ref for Tract library.

This module implements morphisms in the category Ref, whose objects are
flat tuples of positive integers and whose morphisms are refinements
presented by nested tuples. A morphism is recorded by a nested tuple X and
points

    flat(X) ↠ ρ(X),

from the flattening of X to its depth-1 reduction
ρ(X) = (size(X₁),...,size(X_r)), the flat tuple of top-level mode products.
Thus each entry of the codomain is refined by an arbitrary nested
factorization, generalizing the category Fact, whose morphisms are exactly
the Ref morphisms with flat top-level modes (depth ≤ 2 nested tuples).
Composition grafts the finer refinement's top-level modes into the leaves
of the coarser one, so the composite records the full tower of refinements
as a deeper tree.

Kept separate from categories.py for now while the theory is developed.
"""

from typing import Tuple

from .categories import NestedTuple, TupleMorphism
from .fact_morphism import FactMorphism


# *************************************************************************
# THE CATEGORY Ref
# *************************************************************************


class RefMorphism:
    """
    Morphisms in the category Ref.

    A morphism is presented by a nested tuple X = (X₁,...,X_r) of positive
    integers, regarded as the refinement

        flat(X) ↠ ρ(X) = (size(X₁),...,size(X_r)),

    whose source is the flattening of X and whose target is the depth-1
    reduction of X. Both source and target are flat tuples; the nesting of
    X is the morphism datum, with top-level mode i recording the
    factorization tree of the i-th codomain entry.

    The identity on (t₁,...,t_r) is the flat nested tuple (t₁,...,t_r), and
    composition grafts: for f: U ↠ T and g: T ↠ S, the leaves of g's nested
    tuple are the entries of T, and the composite substitutes the i-th
    top-level mode of f for the i-th leaf of g.

    Fact morphisms are exactly the Ref morphisms whose top-level modes are
    flat, and flattening the top-level modes is a functor Ref → Fact (see
    to_fact_morphism).

    The presenting nested tuple may be given as a NestedTuple or as raw
    int/tuple data.
    """

    def __init__(self, nest, name: str = ""):
        if not isinstance(nest, NestedTuple):
            nest = NestedTuple(nest)
        self.nest = nest
        self.name = name
        self._validate_inputs()
        self.modes = tuple(nest.mode(i) for i in range(1, nest.rank() + 1))
        self.domain = nest.flatten()
        self.codomain = tuple(mode.size() for mode in self.modes)

    def _validate_inputs(self) -> None:
        """Verify that the input data defines a valid morphism in the Ref category."""
        for entry in self.nest.flatten():
            if not isinstance(entry, int) or entry < 1:
                raise ValueError(
                    f"Entries must be positive integers, got {entry}"
                )

    def _top_data(self) -> tuple:
        """
        The tuple of top-level mode data, normalizing the int-vs-singleton
        ambiguity of NestedTuple at the root. One entry per codomain entry.
        """
        return tuple(mode.data for mode in self.modes)

    def __repr__(self):
        return f"RefMorphism(nest={self.nest!r})"

    def __str__(self):
        return f"{self.domain} --{self.nest}--> {self.codomain}"

    def __eq__(self, other):
        """
        Structural equality on the tuple of top-level modes; names are
        ignored. The domain and codomain are derived from the modes, so
        this compares the full morphism data.
        """
        if not isinstance(other, RefMorphism):
            return NotImplemented
        return self._top_data() == other._top_data()

    def __hash__(self):
        return hash(self._top_data())

    def size(self) -> int:
        """
        Product of domain entries. Morphisms in Ref preserve size, so
        size() == cosize() always.
        """
        return self.nest.size()

    def cosize(self) -> int:
        """
        Product of codomain entries. Morphisms in Ref preserve size, so
        size() == cosize() always.
        """
        return self.nest.size()

    @classmethod
    def identity(cls, codomain: Tuple[int], name: str = "") -> "RefMorphism":
        """
        Identity morphism on (t₁,...,t_r), presented by the flat nested
        tuple (t₁,...,t_r).
        """
        return cls(NestedTuple(tuple(codomain)), name)

    def is_identity(self) -> bool:
        """Check if the morphism is an identity, i.e. the nested tuple is flat."""
        return self.nest.depth() <= 1

    def are_composable(self, g: "RefMorphism") -> bool:
        """Check if morphisms are composable."""
        return self.codomain == g.domain

    def compose(self, g: "RefMorphism") -> "RefMorphism":
        """
        Compute composition g ∘ f, where f = self, by grafting.

        If f: U ↠ T and g: T ↠ S, the leaves of g's nested tuple are the
        entries of T = ρ(f.nest), so the i-th leaf of g equals the size of
        the i-th top-level mode of f. The composite substitutes that mode
        (as a subtree) for the leaf, giving a nested tuple with flattening
        U and depth-1 reduction S.
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        # Root-normalize g's nest to its tuple of top-level modes, so that a
        # bare-int root (rank-1) grafts into a mode rather than the root.
        return RefMorphism(NestedTuple(g._top_data()).sub(self._top_data()))

    def sum(self, g: "RefMorphism") -> "RefMorphism":
        """Compute sum f ⊕ g, concatenating the top-level modes."""
        return RefMorphism(NestedTuple(self._top_data() + g._top_data()))

    def to_fact_morphism(self) -> FactMorphism:
        """
        The underlying Fact morphism, obtained by flattening each top-level
        mode. This is a functor Ref → Fact: it preserves identities,
        composition, and sums, and is the identity on objects.
        """
        return FactMorphism(
            self.domain,
            self.codomain,
            tuple(mode.flatten() for mode in self.modes),
        )

    @classmethod
    def from_fact_morphism(cls, f: FactMorphism, name: str = "") -> "RefMorphism":
        """
        The Ref morphism presented by a Fact morphism's modes, a depth ≤ 2
        nested tuple. Satisfies from_fact_morphism(f).to_fact_morphism() == f.
        """
        return cls(NestedTuple(f.modes), name)

    # Deprecated aliases
    to_Fact_morphism = to_fact_morphism
    from_Fact_morphism = from_fact_morphism

    def refined_codomain(self) -> NestedTuple:
        """
        The nested tuple presenting the morphism, which refines the
        codomain: refined_codomain().refines(NestedTuple(codomain)) always
        holds.
        """
        return self.nest

    @classmethod
    def from_refinement(
        cls, refined: NestedTuple, coarse: NestedTuple, name: str = ""
    ) -> "RefMorphism":
        """
        Build the Ref morphism flat(refined) ↠ flat(coarse) from a
        refinement refined ↠ coarse with coarse flat. Unlike
        FactMorphism.from_refinement, the relative modes may be nested.
        """
        if coarse.depth() > 1:
            raise ValueError(f"Coarse nested tuple {coarse} must be flat.")
        if not refined.refines(coarse):
            raise ValueError(f"{refined} does not refine {coarse}.")

        return cls(refined.relative_flattening(coarse), name)

    def pullback(self, f: TupleMorphism) -> TupleMorphism:
        """
        Pull back a Tuple morphism f: S → T along self: T′ ↠ T, viewing
        self as the refinement of T = codomain by T′ = domain. Delegates to
        the underlying Fact morphism, since the pulled-back Tuple morphism
        only sees the flattened modes.
        """
        return self.to_fact_morphism().pullback(f)

    def pullback_with_refinement(self, f: TupleMorphism):
        """
        Pull back a Tuple morphism f: S → T along self: T′ ↠ T, returning
        both the induced refinement of the domain and the pulled-back
        morphism.

        The pullback square is

            S′ --f′--> T′
            ↓r         ↓self
            S ---f---> T

        where r: S′ ↠ S is the Ref morphism refining each domain entry sᵢ
        of f with α(i) = j ≠ * by the j-th top-level mode of self (as a
        subtree), and leaving entries with α(i) = * unrefined. Returns the
        pair (r, f′).
        """
        pullback = self.pullback(f)
        top_data = tuple(
            self.modes[j - 1].data if j != 0 else f.domain[i]
            for i, j in enumerate(f.map)
        )
        refinement = RefMorphism(NestedTuple(top_data))
        return refinement, pullback

    def pushforward(self, f: TupleMorphism) -> TupleMorphism:
        """
        Push forward a Tuple morphism f: U → V along self: U′ ↠ U, viewing
        self as the refinement of U = codomain by U′ = domain. Delegates to
        the underlying Fact morphism, since the pushed-forward Tuple
        morphism only sees the flattened modes.
        """
        return self.to_fact_morphism().pushforward(f)

    def pushforward_with_refinement(self, f: TupleMorphism):
        """
        Push forward a Tuple morphism f: U → V along self: U′ ↠ U, returning
        both the induced refinement of the codomain and the pushed-forward
        morphism.

        The pushforward square is

            U′ --f′--> V′
            ↓self      ↓r
            U ---f---> V

        where r: V′ ↠ V is the Ref morphism refining each codomain entry
        v_j in the image of α (say v_j = uᵢ with α(i) = j) by the i-th
        top-level mode of self (as a subtree), and leaving entries outside
        the image unrefined. Returns the pair (r, f′).
        """
        pushforward = self.pushforward(f)
        top_data = tuple(
            self.modes[f.map.index(j)].data if j in f.map else f.codomain[j - 1]
            for j in range(1, len(f.codomain) + 1)
        )
        refinement = RefMorphism(NestedTuple(top_data))
        return refinement, pushforward
