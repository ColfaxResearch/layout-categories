"""
The category CoSpan for Tract library.

This module implements morphisms in the category CoSpan, whose objects are
flat tuples of positive integers and whose morphisms are cospans

    U —f→ X ←b— V

with forward (left) leg f: U → X a Tuple morphism and backward (right) leg
b: X ↠ V a Fact morphism out of the nadir X. Composition is built on the
pushforward of a Tuple morphism along a Fact morphism
(Fact_morphism.pushforward_with_refinement).

Kept separate from categories.py for now while the theory is developed.
"""

from .categories import Tuple_morphism
from .fact_morphism import Fact_morphism


# *************************************************************************
# THE CATEGORY CoSpan
# *************************************************************************


class CoSpan_morphism:
    """
    Morphisms in the category CoSpan.

    A morphism U → V is a cospan U —f→ X ←b— V, where the forward (left) leg
    f: U → X is a Tuple morphism, the backward (right) leg b: X ↠ V is a
    Fact morphism (pointing backward, from the nadir to V), and
    X = f.codomain = b.domain is the nadir. Thus the nadir is a flat
    refinement of V receiving a tuple morphism from U.

    Composition of U —f₁→ X ←b₁— V and V —f₂→ Y ←b₂— W pushes f₂ forward
    along b₁ to a square with corner Y′, then composes the legs:

        U --f₁--> X --f₂′--> Y′
                  ↓b₁        ↓r
                  V --f₂---> Y
                             ↑b₂
                             W

    giving the cospan U —(f₂′ ∘ f₁)→ Y′ ←(b₂ ∘ r)— W (composites written in
    application order). This composition is strictly associative and unital,
    since the chosen pushforward refines each nadir entry by a concatenation
    of blocks.

    :param left: Forward leg, a Tuple morphism U → X
    :type left: Tuple_morphism
    :param right: Backward leg, a Fact morphism X ↠ V
    :type right: Fact_morphism
    :param name: Optional name
    :type name: str
    """

    def __init__(self, left: Tuple_morphism, right: Fact_morphism, name: str = ""):
        self.left = left
        self.right = right
        self.name = name
        self._validate_inputs()
        self.nadir = left.codomain
        self.domain = left.domain
        self.codomain = right.codomain

    def _validate_inputs(self) -> None:
        """
        Verify that the input data defines a valid morphism in the CoSpan
        category.

        :raises ValueError: If morphism is invalid
        """
        if not isinstance(self.left, Tuple_morphism):
            raise ValueError(
                f"Left leg must be a Tuple_morphism, got {type(self.left).__name__}"
            )
        if not isinstance(self.right, Fact_morphism):
            raise ValueError(
                f"Right leg must be a Fact_morphism, got {type(self.right).__name__}"
            )
        if self.left.codomain != self.right.domain:
            raise ValueError(
                f"Legs must share a nadir: left leg has codomain "
                f"{self.left.codomain}, right leg has domain {self.right.domain}"
            )

    def __repr__(self):
        return f"CoSpan_morphism(left={self.left!r}, right={self.right!r})"

    def __str__(self):
        return (
            f"{self.domain} --{self.left.map}--> {self.nadir} "
            f"<--{self.right.modes}-- {self.codomain}"
        )

    def __eq__(self, other):
        """
        Structural equality on both legs; names are ignored. The left leg is
        compared by (domain, codomain, map) since Tuple_morphism does not
        implement structural equality.

        :param other: Object to compare against
        :return: True if other is a CoSpan_morphism with the same legs
        :rtype: bool
        """
        if not isinstance(other, CoSpan_morphism):
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
    def identity(cls, obj, name: str = "") -> "CoSpan_morphism":
        """
        Identity cospan U —id→ U ←id— U on a flat tuple U.

        :param obj: Object to take the identity of
        :type obj: Tuple[int]
        :param name: Optional name
        :type name: str
        :return: Identity cospan
        :rtype: CoSpan_morphism
        """
        return cls(
            Tuple_morphism(obj, obj, tuple(range(1, len(obj) + 1))),
            Fact_morphism.identity(obj),
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

    def are_composable(self, g: "CoSpan_morphism") -> bool:
        """
        Check if cospans are composable.

        :param g: Second cospan
        :type g: CoSpan_morphism
        :return: True if composable
        :rtype: bool
        """
        return self.codomain == g.domain

    def compose(self, g: "CoSpan_morphism") -> "CoSpan_morphism":
        """
        Compute composition g ∘ f, where f = self, by pushing g's forward leg
        forward along f's backward leg and composing legs around the
        resulting square.

        :param g: Second cospan (must have domain = self.codomain)
        :type g: CoSpan_morphism
        :return: The composition g ∘ f
        :rtype: CoSpan_morphism
        :raises ValueError: If not composable
        """
        if self.codomain != g.domain:
            raise ValueError("The given morphisms are not composable.")

        refinement, pushed = self.right.pushforward_with_refinement(g.left)
        return CoSpan_morphism(
            self.left.compose(pushed), refinement.compose(g.right)
        )

    def sum(self, g: "CoSpan_morphism") -> "CoSpan_morphism":
        """
        Compute sum f ⊕ g, taken legwise.

        :param g: Second cospan
        :type g: CoSpan_morphism
        :return: Sum of cospans
        :rtype: CoSpan_morphism
        """
        return CoSpan_morphism(self.left.sum(g.left), self.right.sum(g.right))
