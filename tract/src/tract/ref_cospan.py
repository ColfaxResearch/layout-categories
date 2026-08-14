"""
The category CoSpan(Tuple, Ref) for Tract library.

This module implements morphisms in the category CoSpan(Tuple, Ref), whose
objects are flat tuples of positive integers and whose morphisms are
cospans

    U —f→ X ←b— V

with forward (left) leg f: U → X a Tuple morphism and backward (right) leg
b: X ↠ V a Ref morphism out of the nadir X. The Ref leg is presented by a
nested tuple whose flattening is the nadir and whose depth-1 reduction is
V, so the nadir carries an arbitrary nested refinement of the codomain
receiving a tuple morphism from U. Composition is built on the pushforward
of a Tuple morphism along a Ref morphism
(Ref_morphism.pushforward_with_refinement).

This directly generalizes the category CoSpan (cospan.py): a Fact morphism
is a Ref morphism with flat top-level modes, and flattening the backward
legs (RefCoSpan_morphism.to_CoSpan_morphism) is a functor
CoSpan(Tuple, Ref) → CoSpan.

Kept separate from categories.py for now while the theory is developed.
"""

from .categories import Tuple_morphism
from .cospan import CoSpan_morphism
from .ref_morphism import Ref_morphism


# *************************************************************************
# THE CATEGORY CoSpan(Tuple, Ref)
# *************************************************************************


class RefCoSpan_morphism:
    """
    Morphisms in the category CoSpan(Tuple, Ref).

    A morphism U → V is a cospan U —f→ X ←b— V, where the forward (left)
    leg f: U → X is a Tuple morphism, the backward (right) leg b: X ↠ V is
    a Ref morphism (pointing backward, from the nadir to V), and
    X = f.codomain = b.domain is the nadir. Since b is presented by a
    nested tuple N, the nadir is flat(N) and the codomain is the depth-1
    reduction ρ(N).

    Composition of U —f₁→ X ←b₁— V and V —f₂→ Y ←b₂— W pushes f₂ forward
    along b₁ to a square with corner Y′, then composes the legs:

        U --f₁--> X --f₂′--> Y′
                  ↓b₁        ↓r
                  V --f₂---> Y
                             ↑b₂
                             W

    giving the cospan U —(f₂′ ∘ f₁)→ Y′ ←(b₂ ∘ r)— W (composites written in
    application order). This composition is strictly associative and
    unital, since the chosen pushforward refines each nadir entry by
    grafting subtrees.

    :param left: Forward leg, a Tuple morphism U → X
    :type left: Tuple_morphism
    :param right: Backward leg, a Ref morphism X ↠ V
    :type right: Ref_morphism
    :param name: Optional name
    :type name: str
    """

    def __init__(self, left: Tuple_morphism, right: Ref_morphism, name: str = ""):
        self.left = left
        self.right = right
        self.name = name
        self._validate_inputs()
        self.nadir = left.codomain
        self.domain = left.domain
        self.codomain = right.codomain

    def _validate_inputs(self) -> None:
        """
        Verify that the input data defines a valid morphism in the
        CoSpan(Tuple, Ref) category.

        :raises ValueError: If morphism is invalid
        """
        if not isinstance(self.left, Tuple_morphism):
            raise ValueError(
                f"Left leg must be a Tuple_morphism, got {type(self.left).__name__}"
            )
        if not isinstance(self.right, Ref_morphism):
            raise ValueError(
                f"Right leg must be a Ref_morphism, got {type(self.right).__name__}"
            )
        if self.left.codomain != self.right.domain:
            raise ValueError(
                f"Legs must share a nadir: left leg has codomain "
                f"{self.left.codomain}, right leg has domain {self.right.domain}"
            )

    def __repr__(self):
        return f"RefCoSpan_morphism(left={self.left!r}, right={self.right!r})"

    def __str__(self):
        return (
            f"{self.domain} --{self.left.map}--> {self.nadir} "
            f"<--{self.right.nest}-- {self.codomain}"
        )

    def __eq__(self, other):
        """
        Structural equality on both legs; names are ignored. The left leg
        is compared by (domain, codomain, map) since Tuple_morphism does
        not implement structural equality.

        :param other: Object to compare against
        :return: True if other is a RefCoSpan_morphism with the same legs
        :rtype: bool
        """
        if not isinstance(other, RefCoSpan_morphism):
            return NotImplemented
        return (
            self.right == other.right
            and self.left.domain == other.left.domain
            and self.left.codomain == other.left.codomain
            and self.left.map == other.left.map
        )

    def __hash__(self):
        return hash(
            (self.right, self.left.domain, self.left.codomain, self.left.map)
        )

    @classmethod
    def identity(cls, obj, name: str = "") -> "RefCoSpan_morphism":
        """
        Identity cospan U —id→ U ←id— U on a flat tuple U.

        :param obj: Object to take the identity of
        :type obj: Tuple[int]
        :param name: Optional name
        :type name: str
        :return: Identity cospan
        :rtype: RefCoSpan_morphism
        """
        return cls(
            Tuple_morphism(obj, obj, tuple(range(1, len(obj) + 1))),
            Ref_morphism.identity(obj),
            name,
        )

    def is_identity(self) -> bool:
        """
        Check if the cospan is an identity, i.e. both legs are identities.

        :return: True if identity
        :rtype: bool
        """
        return (
            self.right.is_identity()
            and self.left.domain == self.left.codomain
            and self.left.map == tuple(range(1, len(self.left.domain) + 1))
        )

    def are_composable(self, g: "RefCoSpan_morphism") -> bool:
        """
        Check if cospans are composable.

        :param g: Second cospan
        :type g: RefCoSpan_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain == g.domain

    def compose(self, g: "RefCoSpan_morphism") -> "RefCoSpan_morphism":
        """
        Compute composition g ∘ f, where f = self, by pushing g's forward
        leg forward along f's backward leg and composing legs around the
        resulting square.

        :param g: Second cospan (must have domain = self.codomain)
        :type g: RefCoSpan_morphism
        :return: The composition g ∘ f
        :rtype: RefCoSpan_morphism
        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        refinement, pushed = self.right.pushforward_with_refinement(g.left)
        return RefCoSpan_morphism(
            self.left.compose(pushed), refinement.compose(g.right)
        )

    def sum(self, g: "RefCoSpan_morphism") -> "RefCoSpan_morphism":
        """
        Compute sum f ⊕ g, taken legwise.

        :param g: Second cospan
        :type g: RefCoSpan_morphism
        :return: Sum of cospans
        :rtype: RefCoSpan_morphism
        """
        return RefCoSpan_morphism(self.left.sum(g.left), self.right.sum(g.right))

    def to_CoSpan_morphism(self) -> CoSpan_morphism:
        """
        The underlying CoSpan morphism, obtained by flattening the backward
        leg's top-level modes to a Fact morphism. This is a functor
        CoSpan(Tuple, Ref) → CoSpan: it preserves identities, composition,
        and sums, and is the identity on objects.

        :return: CoSpan morphism with the same nadir, domain, and codomain
        :rtype: CoSpan_morphism
        """
        return CoSpan_morphism(self.left, self.right.to_Fact_morphism())

    @classmethod
    def from_CoSpan_morphism(
        cls, f: CoSpan_morphism, name: str = ""
    ) -> "RefCoSpan_morphism":
        """
        The RefCoSpan morphism presented by a CoSpan morphism, with
        backward leg the depth ≤ 2 Ref morphism of the Fact leg's modes.
        Satisfies from_CoSpan_morphism(f).to_CoSpan_morphism() == f.

        :param f: CoSpan morphism
        :type f: CoSpan_morphism
        :param name: Optional name
        :type name: str
        :return: Corresponding RefCoSpan morphism
        :rtype: RefCoSpan_morphism
        """
        return cls(f.left, Ref_morphism.from_Fact_morphism(f.right), name)
