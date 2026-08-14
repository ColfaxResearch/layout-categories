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

from .categories import NestedTuple, Tuple_morphism
from .fact_morphism import Fact_morphism


# *************************************************************************
# THE CATEGORY Ref
# *************************************************************************


class Ref_morphism:
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
    to_Fact_morphism).

    :param nest: Nested tuple presenting the refinement
    :type nest: NestedTuple, or raw int/tuple data
    :param name: Optional name
    :type name: str
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
        """
        Verify that the input data defines a valid morphism in the Ref
        category.

        :raises ValueError: If morphism is invalid
        """
        for entry in self.nest.flatten():
            if not isinstance(entry, int) or entry < 1:
                raise ValueError(
                    f"Entries must be positive integers, got {entry}"
                )

    def _top_data(self) -> tuple:
        """
        The tuple of top-level mode data, normalizing the int-vs-singleton
        ambiguity of NestedTuple at the root.

        :return: Tuple of raw mode data, one entry per codomain entry
        :rtype: tuple
        """
        return tuple(mode.data for mode in self.modes)

    def __repr__(self):
        return f"Ref_morphism(nest={self.nest!r})"

    def __str__(self):
        return f"{self.domain} --{self.nest}--> {self.codomain}"

    def __eq__(self, other):
        """
        Structural equality on the tuple of top-level modes; names are
        ignored. The domain and codomain are derived from the modes, so
        this compares the full morphism data.

        :param other: Object to compare against
        :return: True if other is a Ref_morphism with the same modes
        :rtype: bool
        """
        if not isinstance(other, Ref_morphism):
            return NotImplemented
        return self._top_data() == other._top_data()

    def __hash__(self):
        return hash(self._top_data())

    def size(self) -> int:
        """
        Product of domain entries. Morphisms in Ref preserve size, so
        size() == cosize() always.

        :return: Size of domain
        :rtype: int
        """
        return self.nest.size()

    def cosize(self) -> int:
        """
        Product of codomain entries. Morphisms in Ref preserve size, so
        size() == cosize() always.

        :return: Size of codomain
        :rtype: int
        """
        return self.nest.size()

    @classmethod
    def identity(cls, codomain: Tuple[int], name: str = "") -> "Ref_morphism":
        """
        Identity morphism on (t₁,...,t_r), presented by the flat nested
        tuple (t₁,...,t_r).

        :param codomain: Object to take the identity of
        :type codomain: Tuple[int]
        :param name: Optional name
        :type name: str
        :return: Identity morphism
        :rtype: Ref_morphism
        """
        return cls(NestedTuple(tuple(codomain)), name)

    def is_identity(self) -> bool:
        """
        Check if the morphism is an identity, i.e. the nested tuple is flat.

        :return: True if identity
        :rtype: bool
        """
        return self.nest.depth() <= 1

    def are_composable(self, g: "Ref_morphism") -> bool:
        """
        Check if morphisms are composable.

        :param g: Second morphism
        :type g: Ref_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain == g.domain

    def compose(self, g: "Ref_morphism") -> "Ref_morphism":
        """
        Compute composition g ∘ f, where f = self, by grafting.

        If f: U ↠ T and g: T ↠ S, the leaves of g's nested tuple are the
        entries of T = ρ(f.nest), so the i-th leaf of g equals the size of
        the i-th top-level mode of f. The composite substitutes that mode
        (as a subtree) for the leaf, giving a nested tuple with flattening
        U and depth-1 reduction S.

        :param g: Second morphism (must have domain = self.codomain)
        :type g: Ref_morphism
        :return: The composition g ∘ f
        :rtype: Ref_morphism
        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        # Root-normalize g's nest to its tuple of top-level modes, so that a
        # bare-int root (rank-1) grafts into a mode rather than the root.
        return Ref_morphism(NestedTuple(g._top_data()).sub(self._top_data()))

    def sum(self, g: "Ref_morphism") -> "Ref_morphism":
        """
        Compute sum f ⊕ g, concatenating the top-level modes.

        :param g: Second morphism
        :type g: Ref_morphism
        :return: Sum of morphisms
        :rtype: Ref_morphism
        """
        return Ref_morphism(NestedTuple(self._top_data() + g._top_data()))

    def to_Fact_morphism(self) -> Fact_morphism:
        """
        The underlying Fact morphism, obtained by flattening each top-level
        mode. This is a functor Ref → Fact: it preserves identities,
        composition, and sums, and is the identity on objects.

        :return: Fact morphism with the same domain and codomain
        :rtype: Fact_morphism
        """
        return Fact_morphism(
            self.domain,
            self.codomain,
            tuple(mode.flatten() for mode in self.modes),
        )

    @classmethod
    def from_Fact_morphism(cls, f: Fact_morphism, name: str = "") -> "Ref_morphism":
        """
        The Ref morphism presented by a Fact morphism's modes, a depth ≤ 2
        nested tuple. Satisfies from_Fact_morphism(f).to_Fact_morphism() == f.

        :param f: Fact morphism
        :type f: Fact_morphism
        :param name: Optional name
        :type name: str
        :return: Corresponding Ref morphism
        :rtype: Ref_morphism
        """
        return cls(NestedTuple(f.modes), name)

    def refined_codomain(self) -> NestedTuple:
        """
        The nested tuple presenting the morphism, which refines the
        codomain: refined_codomain().refines(NestedTuple(codomain)) always
        holds.

        :return: The presenting nested tuple
        :rtype: NestedTuple
        """
        return self.nest

    @classmethod
    def from_refinement(
        cls, refined: NestedTuple, coarse: NestedTuple, name: str = ""
    ) -> "Ref_morphism":
        """
        Build the Ref morphism flat(refined) ↠ flat(coarse) from a
        refinement refined ↠ coarse with coarse flat. Unlike
        Fact_morphism.from_refinement, the relative modes may be nested.

        :param refined: Refining nested tuple
        :type refined: NestedTuple
        :param coarse: Refined nested tuple (must be flat)
        :type coarse: NestedTuple
        :param name: Optional name
        :type name: str
        :return: Corresponding Ref morphism
        :rtype: Ref_morphism
        :raises ValueError: If refined does not refine coarse, or coarse is
            not flat
        """
        if coarse.depth() > 1:
            raise ValueError(f"Coarse nested tuple {coarse} must be flat.")
        if not refined.refines(coarse):
            raise ValueError(f"{refined} does not refine {coarse}.")

        return cls(refined.relative_flattening(coarse), name)

    def pullback(self, f: Tuple_morphism) -> Tuple_morphism:
        """
        Pull back a Tuple morphism f: S → T along self: T′ ↠ T, viewing
        self as the refinement of T = codomain by T′ = domain. Delegates to
        the underlying Fact morphism, since the pulled-back Tuple morphism
        only sees the flattened modes.

        :param f: Tuple morphism with codomain equal to self.codomain
        :type f: Tuple_morphism
        :return: The pullback of f along self
        :rtype: Tuple_morphism
        :raises ValueError: If f's codomain does not match
        """
        return self.to_Fact_morphism().pullback(f)

    def pullback_with_refinement(self, f: Tuple_morphism):
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
        subtree), and leaving entries with α(i) = * unrefined.

        :param f: Tuple morphism with codomain equal to self.codomain
        :type f: Tuple_morphism
        :return: Pair (r, f′) of the induced Ref morphism r: S′ ↠ S and the
            pullback f′: S′ → T′
        :rtype: tuple[Ref_morphism, Tuple_morphism]
        :raises ValueError: If f's codomain does not match
        """
        pullback = self.pullback(f)
        top_data = tuple(
            self.modes[j - 1].data if j != 0 else f.domain[i]
            for i, j in enumerate(f.map)
        )
        refinement = Ref_morphism(NestedTuple(top_data))
        return refinement, pullback

    def pushforward(self, f: Tuple_morphism) -> Tuple_morphism:
        """
        Push forward a Tuple morphism f: U → V along self: U′ ↠ U, viewing
        self as the refinement of U = codomain by U′ = domain. Delegates to
        the underlying Fact morphism, since the pushed-forward Tuple
        morphism only sees the flattened modes.

        :param f: Tuple morphism with domain equal to self.codomain
        :type f: Tuple_morphism
        :return: The pushforward of f along self
        :rtype: Tuple_morphism
        :raises ValueError: If f's domain does not match
        """
        return self.to_Fact_morphism().pushforward(f)

    def pushforward_with_refinement(self, f: Tuple_morphism):
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
        the image unrefined.

        :param f: Tuple morphism with domain equal to self.codomain
        :type f: Tuple_morphism
        :return: Pair (r, f′) of the induced Ref morphism r: V′ ↠ V and the
            pushforward f′: U′ → V′
        :rtype: tuple[Ref_morphism, Tuple_morphism]
        :raises ValueError: If f's domain does not match
        """
        pushforward = self.pushforward(f)
        top_data = tuple(
            self.modes[f.map.index(j)].data if j in f.map else f.codomain[j - 1]
            for j in range(1, len(f.codomain) + 1)
        )
        refinement = Ref_morphism(NestedTuple(top_data))
        return refinement, pushforward
