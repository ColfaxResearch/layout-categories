"""
The category Span(Tuple, Ref) for Tract library.

This module implements morphisms in the category Span(Tuple, Ref), whose
objects are flat tuples of positive integers and whose morphisms are spans

    U ←b— X —f→ V

with backward (left) leg b: X ↠ U a Ref morphism and forward (right) leg
f: X → V a Tuple morphism, sharing the apex X. The Ref leg is presented by
a nested tuple whose flattening is the apex and whose depth-1 reduction is
U, so the apex carries an arbitrary nested factorization of the domain.
Composition is built on the pullback of a Tuple morphism along a Ref
morphism (Ref_morphism.pullback_with_refinement).

This directly generalizes the category Span (span.py): a Fact morphism is
a Ref morphism with flat top-level modes, and flattening the backward legs
(RefSpan_morphism.to_Span_morphism) is a functor Span(Tuple, Ref) → Span.

Kept separate from categories.py for now while the theory is developed.
"""

from .categories import Tuple_morphism
from .ref_morphism import Ref_morphism
from .span import Span_morphism


# *************************************************************************
# THE CATEGORY Span(Tuple, Ref)
# *************************************************************************


class RefSpan_morphism:
    """
    Morphisms in the category Span(Tuple, Ref).

    A morphism U → V is a span U ←b— X —f→ V, where the backward (left) leg
    b: X ↠ U is a Ref morphism with codomain U, the forward (right) leg
    f: X → V is a Tuple morphism, and X = b.domain = f.domain is the apex.
    Since b is presented by a nested tuple N, the apex is flat(N) and the
    domain is the depth-1 reduction ρ(N).

    Composition of U ←b₁— X —f₁→ V and V ←b₂— Y —f₂→ W pulls f₁ back along
    b₂ to a square with corner X′, then composes the legs:

        X′ --f₁′--> Y --f₂--> W
        ↓r          ↓b₂
        X --f₁----> V
        ↓b₁
        U

    giving the span U ←(b₁ ∘ r)— X′ —(f₂ ∘ f₁′)→ W. This composition is
    strictly associative and unital, since the chosen pullback refines each
    apex entry by grafting subtrees.

    :param left: Backward leg, a Ref morphism X ↠ U
    :type left: Ref_morphism
    :param right: Forward leg, a Tuple morphism X → V
    :type right: Tuple_morphism
    :param name: Optional name
    :type name: str
    """

    def __init__(self, left: Ref_morphism, right: Tuple_morphism, name: str = ""):
        self.left = left
        self.right = right
        self.name = name
        self._validate_inputs()
        self.apex = left.domain
        self.domain = left.codomain
        self.codomain = right.codomain

    def _validate_inputs(self) -> None:
        """
        Verify that the input data defines a valid morphism in the
        Span(Tuple, Ref) category.

        :raises ValueError: If morphism is invalid
        """
        if not isinstance(self.left, Ref_morphism):
            raise ValueError(
                f"Left leg must be a Ref_morphism, got {type(self.left).__name__}"
            )
        if not isinstance(self.right, Tuple_morphism):
            raise ValueError(
                f"Right leg must be a Tuple_morphism, got {type(self.right).__name__}"
            )
        if self.left.domain != self.right.domain:
            raise ValueError(
                f"Legs must share an apex: left leg has domain "
                f"{self.left.domain}, right leg has domain {self.right.domain}"
            )

    def __repr__(self):
        return f"RefSpan_morphism(left={self.left!r}, right={self.right!r})"

    def __str__(self):
        return (
            f"{self.domain} <--{self.left.nest}-- {self.apex} "
            f"--{self.right.map}--> {self.codomain}"
        )

    def __eq__(self, other):
        """
        Structural equality on both legs; names are ignored. The right leg
        is compared by (domain, codomain, map) since Tuple_morphism does
        not implement structural equality.

        :param other: Object to compare against
        :return: True if other is a RefSpan_morphism with the same legs
        :rtype: bool
        """
        if not isinstance(other, RefSpan_morphism):
            return NotImplemented
        return (
            self.left == other.left
            and self.right.domain == other.right.domain
            and self.right.codomain == other.right.codomain
            and self.right.map == other.right.map
        )

    def __hash__(self):
        return hash(
            (self.left, self.right.domain, self.right.codomain, self.right.map)
        )

    @classmethod
    def identity(cls, obj, name: str = "") -> "RefSpan_morphism":
        """
        Identity span U ←id— U —id→ U on a flat tuple U.

        :param obj: Object to take the identity of
        :type obj: Tuple[int]
        :param name: Optional name
        :type name: str
        :return: Identity span
        :rtype: RefSpan_morphism
        """
        return cls(
            Ref_morphism.identity(obj),
            Tuple_morphism(obj, obj, tuple(range(1, len(obj) + 1))),
            name,
        )

    def is_identity(self) -> bool:
        """
        Check if the span is an identity, i.e. both legs are identities.

        :return: True if identity
        :rtype: bool
        """
        return (
            self.left.is_identity()
            and self.right.domain == self.right.codomain
            and self.right.map == tuple(range(1, len(self.right.domain) + 1))
        )

    def are_composable(self, g: "RefSpan_morphism") -> bool:
        """
        Check if spans are composable.

        :param g: Second span
        :type g: RefSpan_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain == g.domain

    def compose(self, g: "RefSpan_morphism") -> "RefSpan_morphism":
        """
        Compute composition g ∘ f, where f = self, by pulling f's forward
        leg back along g's backward leg and composing legs around the
        resulting square.

        :param g: Second span (must have domain = self.codomain)
        :type g: RefSpan_morphism
        :return: The composition g ∘ f
        :rtype: RefSpan_morphism
        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        refinement, pulled = g.left.pullback_with_refinement(self.right)
        return RefSpan_morphism(
            refinement.compose(self.left), pulled.compose(g.right)
        )

    def sum(self, g: "RefSpan_morphism") -> "RefSpan_morphism":
        """
        Compute sum f ⊕ g, taken legwise.

        :param g: Second span
        :type g: RefSpan_morphism
        :return: Sum of spans
        :rtype: RefSpan_morphism
        """
        return RefSpan_morphism(self.left.sum(g.left), self.right.sum(g.right))

    def to_Span_morphism(self) -> Span_morphism:
        """
        The underlying Span morphism, obtained by flattening the backward
        leg's top-level modes to a Fact morphism. This is a functor
        Span(Tuple, Ref) → Span: it preserves identities, composition, and
        sums, and is the identity on objects.

        :return: Span morphism with the same apex, domain, and codomain
        :rtype: Span_morphism
        """
        return Span_morphism(self.left.to_Fact_morphism(), self.right)

    @classmethod
    def from_Span_morphism(cls, f: Span_morphism, name: str = "") -> "RefSpan_morphism":
        """
        The RefSpan morphism presented by a Span morphism, with backward
        leg the depth ≤ 2 Ref morphism of the Fact leg's modes. Satisfies
        from_Span_morphism(f).to_Span_morphism() == f.

        :param f: Span morphism
        :type f: Span_morphism
        :param name: Optional name
        :type name: str
        :return: Corresponding RefSpan morphism
        :rtype: RefSpan_morphism
        """
        return cls(Ref_morphism.from_Fact_morphism(f.left), f.right, name)
