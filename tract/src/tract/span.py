"""
The category Span for Tract library.

This module implements morphisms in the category Span, whose objects are
flat tuples of positive integers and whose morphisms are spans

    U ←b— X —f→ V

with backward (left) leg b: X ↠ U a Fact morphism and forward (right) leg
f: X → V a Tuple morphism, sharing the apex X. Composition is built on the
pullback of a Tuple morphism along a Fact morphism
(Fact_morphism.pullback_with_refinement).

Kept separate from categories.py for now while the theory is developed.
"""

from .categories import Tuple_morphism
from .fact_morphism import Fact_morphism


# *************************************************************************
# THE CATEGORY Span
# *************************************************************************


class Span_morphism:
    """
    Morphisms in the category Span.

    A morphism U → V is a span U ←b— X —f→ V, where the backward (left) leg
    b: X ↠ U is a Fact morphism with codomain U, the forward (right) leg
    f: X → V is a Tuple morphism, and X = b.domain = f.domain is the apex.

    Composition of U ←b₁— X —f₁→ V and V ←b₂— Y —f₂→ W pulls f₁ back along
    b₂ to a square with corner X′, then composes the legs:

        X′ --f₁′--> Y --f₂--> W
        ↓r          ↓b₂
        X --f₁----> V
        ↓b₁
        U

    giving the span U ←(b₁ ∘ r)— X′ —(f₂ ∘ f₁′)→ W. This composition is
    strictly associative and unital, since the chosen pullback refines each
    apex entry by a concatenation of blocks.

    :param left: Backward leg, a Fact morphism X ↠ U
    :type left: Fact_morphism
    :param right: Forward leg, a Tuple morphism X → V
    :type right: Tuple_morphism
    :param name: Optional name
    :type name: str
    """

    def __init__(self, left: Fact_morphism, right: Tuple_morphism, name: str = ""):
        self.left = left
        self.right = right
        self.name = name
        self._validate_inputs()
        self.apex = left.domain
        self.domain = left.codomain
        self.codomain = right.codomain

    def _validate_inputs(self) -> None:
        """
        Verify that the input data defines a valid morphism in the Span
        category.

        :raises ValueError: If morphism is invalid
        """
        if not isinstance(self.left, Fact_morphism):
            raise ValueError(
                f"Left leg must be a Fact_morphism, got {type(self.left).__name__}"
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
        return f"Span_morphism(left={self.left!r}, right={self.right!r})"

    def __str__(self):
        return (
            f"{self.domain} <--{self.left.modes}-- {self.apex} "
            f"--{self.right.map}--> {self.codomain}"
        )

    def __eq__(self, other):
        """
        Structural equality on both legs; names are ignored. The right leg is
        compared by (domain, codomain, map) since Tuple_morphism does not
        implement structural equality.

        :param other: Object to compare against
        :return: True if other is a Span_morphism with the same legs
        :rtype: bool
        """
        if not isinstance(other, Span_morphism):
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
    def identity(cls, obj, name: str = "") -> "Span_morphism":
        """
        Identity span U ←id— U —id→ U on a flat tuple U.

        :param obj: Object to take the identity of
        :type obj: Tuple[int]
        :param name: Optional name
        :type name: str
        :return: Identity span
        :rtype: Span_morphism
        """
        return cls(
            Fact_morphism.identity(obj),
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

    def are_composable(self, g: "Span_morphism") -> bool:
        """
        Check if spans are composable.

        :param g: Second span
        :type g: Span_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain == g.domain

    def compose(self, g: "Span_morphism") -> "Span_morphism":
        """
        Compute composition g ∘ f, where f = self, by pulling f's forward leg
        back along g's backward leg and composing legs around the resulting
        square.

        :param g: Second span (must have domain = self.codomain)
        :type g: Span_morphism
        :return: The composition g ∘ f
        :rtype: Span_morphism
        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        refinement, pulled = g.left.pullback_with_refinement(self.right)
        return Span_morphism(
            refinement.compose(self.left), pulled.compose(g.right)
        )

    def sum(self, g: "Span_morphism") -> "Span_morphism":
        """
        Compute sum f ⊕ g, taken legwise.

        :param g: Second span
        :type g: Span_morphism
        :return: Sum of spans
        :rtype: Span_morphism
        """
        return Span_morphism(self.left.sum(g.left), self.right.sum(g.right))
